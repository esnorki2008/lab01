"""Preparación de datos, pronóstico, pedido sugerido y publicación para un corte.

Flujo por corte T (cada paso idempotente, escribe a ``processed/staging``):

    prepare_features -> generate_forecast -> build_orders -> publish

``publish`` copia a ``data/forecasts/cutoff=T`` (inmutable). Si ya existe un
pronóstico publicado distinto para T (p. ej. código nuevo o re-expresión con
datos corregidos) el original NO se pisa: el nuevo va a ``forecasts/restated``.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .features import build_panel, cutoff_features
from .io_utils import (
    code_fingerprint,
    code_is_dirty,
    code_version,
    frame_hash,
    read_json,
    sha256_file,
    write_csv,
    write_json,
    write_parquet,
)
from .lake import build_snapshot
from .models import load_model, make_model, qcol
from .ordering import suggest_orders
from .training import PANEL_HISTORY_DAYS, champion_entry

log = logging.getLogger(__name__)


class DataQualityError(RuntimeError):
    """Los datos disponibles no permiten emitir un pronóstico confiable."""


def _key(cutoff: pd.Timestamp) -> str:
    return f"cutoff={pd.Timestamp(cutoff):%Y-%m-%d}"


def staging_dir(settings: Settings, cutoff, restate: bool = False, root: Path | None = None) -> Path:
    base = root or settings.path("processed_dir") / "staging"
    return base / ("restated" if restate else "official") / _key(cutoff)


def readiness_report(settings: Settings, snap, cutoff: pd.Timestamp) -> dict:
    v = settings["validation"]
    sales = snap.sales
    dates = pd.to_datetime(sales["date"])
    last = dates.max()
    staleness = int((cutoff - last).days) if pd.notna(last) else 10**6
    recent = sales[dates >= cutoff - pd.Timedelta(days=56)]
    active = recent.groupby(["store_nbr", "family"])["sales"].sum()
    active = set(active[active > 0].index)
    last7 = sales[dates >= cutoff - pd.Timedelta(days=7)]
    seen = set(map(tuple, last7[["store_nbr", "family"]].drop_duplicates().to_numpy().tolist()))
    coverage = len(active & seen) / len(active) if active else 0.0
    prev_day = sales[dates == cutoff - pd.Timedelta(days=1)]
    got_prev = set(map(tuple, prev_day[["store_nbr", "family"]].drop_duplicates().to_numpy().tolist()))
    missing_prev = 1 - len(active & got_prev) / len(active) if active else 0.0
    rep = {"last_sales_date": None if pd.isna(last) else f"{last:%Y-%m-%d}", "staleness_days": staleness,
           "active_series": len(active), "coverage_last_7d": round(coverage, 4),
           "missing_last_day_frac": round(missing_prev, 4), "warnings": []}
    if 0 < missing_prev < 1 and staleness <= 1:
        rep["warnings"].append(f"entrega parcial: falta el último día en {missing_prev:.0%} de las series "
                               "(se imputan con su nivel reciente)")
    if staleness > 1:
        rep["warnings"].append(f"ventas con {staleness} días de atraso (se pronostica con lo disponible)")
    if staleness > int(v["max_sales_staleness_days"]):
        raise DataQualityError(f"Ventas desactualizadas: último dato {rep['last_sales_date']} "
                               f"({staleness} días > {v['max_sales_staleness_days']}). No se publica.")
    if coverage < float(v["min_series_coverage"]):
        raise DataQualityError(f"Cobertura insuficiente en los últimos 7 días: {coverage:.0%}")
    return rep


def chain_closure_days(sales: pd.DataFrame) -> list[str]:
    """Días "MM-DD" en que toda la cadena estuvo cerrada en la historia (sin filas o venta total 0),
    p. ej. 25 de diciembre. Se infieren de los datos, no se escriben a mano."""
    if sales.empty:
        return []
    d = pd.to_datetime(sales["date"])
    total = sales.groupby(d)["sales"].sum()
    full = pd.date_range(total.index.min(), total.index.max(), freq="D")
    total = total.reindex(full)
    closed = total[(total.isna()) | (total <= 0)].index
    return sorted({f"{x:%m-%d}" for x in closed})


def prepare_features(settings: Settings, cutoff, restate: bool = False, root: Path | None = None) -> dict:
    cutoff = pd.Timestamp(cutoff).normalize()
    snap = build_snapshot(settings, cutoff, include_later_deliveries=restate)
    report = readiness_report(settings, snap, cutoff)
    for w in report["warnings"]:
        log.warning(w)
    h = int(settings["forecast"]["horizon_days"])
    tx_lag = int(settings["source_system"]["sources"]["transactions"]["lag_days"])
    panel = build_panel(snap, end=cutoff + pd.Timedelta(days=h - 1),
                        start=cutoff - pd.Timedelta(days=PANEL_HISTORY_DAYS + 56))
    feats = cutoff_features(panel, cutoff, h, tx_lag)
    out = staging_dir(settings, cutoff, restate, root)
    write_parquet(feats, out / "features.parquet")
    manifest = {"cutoff": f"{cutoff:%Y-%m-%d}", "restate": restate, "readiness": report,
                "closure_days": chain_closure_days(snap.sales),
                "inputs": snap.manifest(settings.root), "features_hash": frame_hash(feats)}
    write_json(manifest, out / "inputs.json")
    # copia "procesada" versionada con DVC (features con los que se emitió cada pronóstico)
    if root is None and not restate:
        write_parquet(feats, settings.path("processed_dir") / "features" / f"{_key(cutoff)}.parquet")
    return {"rows": len(feats), **report}


def _load_champion(settings: Settings, cutoff: pd.Timestamp):
    entry = champion_entry(settings, cutoff)
    if entry and entry.get("model_path"):
        path = settings.path("models_dir") / entry["model_path"]
        return load_model(path), entry, sha256_file(path)
    name = entry["champion"] if entry else settings["model_selection"]["initial_champion"]
    return make_model(name, settings), entry, None


def generate_forecast(settings: Settings, cutoff, restate: bool = False, root: Path | None = None) -> dict:
    cutoff = pd.Timestamp(cutoff).normalize()
    out = staging_dir(settings, cutoff, restate, root)
    feats = pd.read_parquet(out / "features.parquet")
    model, entry, model_sha = _load_champion(settings, cutoff)
    pred = model.predict(feats)
    fc = pd.concat([feats[["cutoff", "date", "h", "store_nbr", "family"]], pred], axis=1)
    qc = [qcol(q) for q in settings["forecast"]["quantiles"]]
    # días de cierre de toda la cadena (observados en años anteriores): demanda 0, no se pide para ellos
    closures = set((read_json(out / "inputs.json") or {}).get("closure_days", []))
    if closures:
        closed = pd.to_datetime(fc["date"]).dt.strftime("%m-%d").isin(closures)
        fc.loc[closed, qc] = 0.0
    fc[qc] = fc[qc].round(4)
    fc["model"] = model.name
    write_parquet(fc, out / "forecast.parquet")
    info = {"model": model.name, "model_trained_cutoff": entry and entry.get("trained_cutoff"),
            "model_path": entry and entry.get("model_path"), "model_sha256": model_sha}
    write_json(info, out / "model.json")
    return info


def build_orders(settings: Settings, cutoff, restate: bool = False, root: Path | None = None) -> dict:
    cutoff = pd.Timestamp(cutoff).normalize()
    out = staging_dir(settings, cutoff, restate, root)
    fc = pd.read_parquet(out / "forecast.parquet")
    feats = pd.read_parquet(out / "features.parquet", columns=["store_nbr", "family", "h", "mean_7"])
    orders = suggest_orders(fc, settings)
    naive = feats[feats["h"] == 0].set_index(["store_nbr", "family"])["mean_7"]
    days = int(settings["business"]["review_period_days"])
    orders["naive_order_qty"] = (orders.set_index(["store_nbr", "family"]).index.map(naive)
                                 .to_numpy(dtype=float) * days)
    orders["naive_order_qty"] = np.nan_to_num(orders["naive_order_qty"]).round(2)
    write_parquet(orders, out / "orders.parquet")
    return {"orders": len(orders), "total_qty": float(orders["order_qty"].sum())}


def _artifact_hashes(d: Path) -> dict:
    return {"forecast": frame_hash(pd.read_parquet(d / "forecast.parquet")),
            "orders": frame_hash(pd.read_parquet(d / "orders.parquet"))}


def publish(settings: Settings, cutoff, restate: bool = False) -> dict:
    cutoff = pd.Timestamp(cutoff).normalize()
    src = staging_dir(settings, cutoff, restate)
    hashes = _artifact_hashes(src)
    official = settings.path("forecasts_dir") / _key(cutoff)
    target = settings.path("forecasts_dir") / "restated" / _key(cutoff) if restate else official
    if code_is_dirty(settings.root):
        log.warning("Se publica con cambios sin commitear en src/dags/config: el commit registrado no "
                    "basta para reproducir; use code_fingerprint del metadata.")
    if not restate and (official / "metadata.json").exists():
        prev = read_json(official / "metadata.json")
        if prev.get("hashes") == hashes:
            log.info("Pronóstico %s ya publicado con idéntico contenido (no-op).", cutoff.date())
            return {**prev, "status": "unchanged"}
        log.warning("Ya existe un pronóstico publicado distinto para %s; el original se conserva y el "
                    "nuevo se guarda como re-expresión.", cutoff.date())
        target = settings.path("forecasts_dir") / "restated" / _key(cutoff)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("forecast.parquet", "orders.parquet"):
        shutil.copyfile(src / name, target / name)
    orders = pd.read_parquet(src / "orders.parquet")
    write_csv(orders, target / "orders.csv")
    inputs = read_json(src / "inputs.json")
    meta = {
        "cutoff": f"{cutoff:%Y-%m-%d}",
        "kind": "restated" if target != official else "official",
        "hashes": hashes,
        "model": read_json(src / "model.json"),
        "code_version": code_version(settings.root),
        "code_dirty": code_is_dirty(settings.root),
        "code_fingerprint": code_fingerprint(settings.root),
        "config_hash": settings.config_hash,
        "business_params": settings["business"],
        "readiness": inputs["readiness"],
        "features_hash": inputs["features_hash"],
        "n_input_files": len(inputs["inputs"]),
    }
    write_json(inputs["inputs"], target / "inputs_manifest.json")
    write_json(meta, target / "metadata.json")
    log.info("Publicado %s en %s", cutoff.date(), target)
    return {**meta, "status": "published", "path": str(target)}


def run_cutoff(settings: Settings, cutoff, restate: bool = False, root: Path | None = None) -> dict:
    prepare_features(settings, cutoff, restate, root)
    generate_forecast(settings, cutoff, restate, root)
    build_orders(settings, cutoff, restate, root)
    return _artifact_hashes(staging_dir(settings, cutoff, restate, root))


def reproduce(settings: Settings, cutoff, workdir: Path) -> dict:
    """Recalcula un pronóstico publicado y verifica que sea idéntico."""
    cutoff = pd.Timestamp(cutoff).normalize()
    meta = read_json(settings.path("forecasts_dir") / _key(cutoff) / "metadata.json")
    if meta is None:
        raise FileNotFoundError(f"No hay pronóstico publicado para {cutoff.date()}")
    hashes = run_cutoff(settings, cutoff, root=workdir)
    return {"cutoff": f"{cutoff:%Y-%m-%d}", "published": meta["hashes"], "reproduced": hashes,
            "identical": hashes == meta["hashes"], "published_code_version": meta["code_version"],
            "current_code_version": code_version(settings.root),
            "same_code": meta.get("code_fingerprint") == code_fingerprint(settings.root)}
