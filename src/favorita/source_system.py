"""Simulador del sistema de origen de la empresa.

A partir de los archivos de Kaggle construye:

* ``data/history/<fuente>.parquet``: carga histórica inicial = todo lo que ya se
  había entregado antes de ``history_end``.
* ``data/inbox/<fuente>/delivered=YYYY-MM-DD/<fuente>_<fecha_dato>_v<k>.csv``:
  un archivo por fecha (y versión) y por fuente, en la carpeta del día en que
  "llega". Las fuentes tienen distinta frecuencia y anticipación (``lag_days``)
  y se inyectan situaciones reales: entregas parciales, correcciones,
  archivos inválidos, atrasos y filas duplicadas.

El simulador NO es parte del producto: representa a la empresa. El flujo
(ingesta) sólo lee la bandeja.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .io_utils import read_json, sha256_file, write_csv, write_json, write_parquet

log = logging.getLogger(__name__)

KAGGLE_FILES = ("train.csv", "stores.csv", "holidays_events.csv", "oil.csv", "transactions.csv")
MANIFEST = "_manifest.json"


# ----------------------------------------------------------------------------
# Lectura tolerante de Kaggle (no limpia: los problemas los detecta la validación)
# ----------------------------------------------------------------------------
def read_kaggle(kaggle_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [f for f in KAGGLE_FILES if not (kaggle_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan archivos de Kaggle en {kaggle_dir}: {missing}. Ver README.md (sección Datos)."
        )
    train = pd.read_csv(kaggle_dir / "train.csv", dtype=str, keep_default_na=False)
    out = {
        "sales": train[["date", "store_nbr", "family", "sales"]].copy(),
        "promotions": train[["date", "store_nbr", "family", "onpromotion"]].copy(),
        "transactions": pd.read_csv(kaggle_dir / "transactions.csv", dtype=str, keep_default_na=False),
        "oil": pd.read_csv(kaggle_dir / "oil.csv", dtype=str, keep_default_na=False),
        "holidays": pd.read_csv(kaggle_dir / "holidays_events.csv", dtype=str, keep_default_na=False),
        "stores": pd.read_csv(kaggle_dir / "stores.csv", dtype=str, keep_default_na=False),
    }
    for name, df in out.items():
        if name != "stores":
            df["_d"] = pd.to_datetime(df["date"], errors="coerce")
    return out


MIN_HISTORY_DAYS = 150  # el modelo necesita ~4 meses de historia (medias de 112 días, rezagos)


def operating_range(sales_dates: pd.Series, store_ids: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Rango de fechas "reales" de ventas, robusto a fechas mal escritas (p. ej. 2071-08-10).

    Una fecha cuenta si trae ventas de al menos la mitad de las tiendas de un día típico.
    """
    counts = store_ids.groupby(sales_dates).nunique()
    if counts.empty:
        raise ValueError("train.csv no tiene fechas válidas")
    ok = counts[counts >= 0.5 * counts.median()]
    bad = counts.index.difference(ok.index)
    outside = [d for d in bad if d > ok.index.max() or d < ok.index.min()]
    if outside:
        log.warning("Se ignoran %d fechas aisladas fuera del rango operativo: %s", len(outside),
                    [f"{d:%Y-%m-%d}" for d in outside[:5]])
    return ok.index.min(), ok.index.max()


def resolve_history_end(min_date: pd.Timestamp, max_date: pd.Timestamp, cfg: dict) -> pd.Timestamp:
    he = cfg.get("history_end", "auto")

    def monday(d: pd.Timestamp) -> pd.Timestamp:
        return (d + pd.Timedelta(days=(7 - d.weekday()) % 7)).normalize()

    floor = monday(min_date + pd.Timedelta(days=MIN_HISTORY_DAYS))
    if he and str(he) != "auto":
        chosen = pd.Timestamp(he)
    else:
        chosen = monday(max_date - pd.Timedelta(days=int(cfg.get("inbox_days", 182))))
    if chosen < floor:
        # Con poca historia (train.csv corto) se acorta la bandeja en vez de dejar al modelo sin datos.
        log.warning("history_end %s deja menos de %d días de historia; se usa %s", chosen.date(),
                    MIN_HISTORY_DAYS, floor.date())
        chosen = floor
    return chosen


def _rng(seed: int, *parts: object) -> np.random.Generator:
    h = hashlib.sha256(("|".join(map(str, (seed, *parts)))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


@dataclass
class InboxFile:
    source: str
    data_date: pd.Timestamp
    version: int
    delivered: pd.Timestamp
    frame: pd.DataFrame
    note: str = ""

    def relpath(self) -> str:
        return (f"{self.source}/delivered={self.delivered:%Y-%m-%d}/"
                f"{self.source}_{self.data_date:%Y-%m-%d}_v{self.version}.csv")


def _data_date(df: pd.DataFrame, granularity: str) -> pd.Series:
    if granularity == "month":
        return df["_d"].dt.to_period("M").dt.to_timestamp()
    return df["_d"]


def _corrupt(frame: pd.DataFrame, source: str, rng: np.random.Generator) -> tuple[pd.DataFrame, str]:
    f = frame.copy()
    value_col = {"sales": "sales", "transactions": "transactions", "oil": "dcoilwtico"}.get(source)
    kind = rng.choice(["negative", "missing_column", "garbage", "wrong_date"])
    if value_col is None:
        kind = "missing_column"
    if kind == "negative":
        idx = rng.choice(len(f), size=max(1, len(f) // 3), replace=False)
        vals = pd.to_numeric(f[value_col], errors="coerce").fillna(1.0)
        vals.iloc[idx] = -vals.iloc[idx].abs() - 1
        f[value_col] = vals.round(3).astype(str)
    elif kind == "missing_column":
        f = f.drop(columns=[f.columns[-1]])
    elif kind == "garbage":
        idx = rng.choice(len(f), size=max(1, len(f) // 2), replace=False)
        f[value_col] = f[value_col].astype(object)
        f.iloc[idx, f.columns.get_loc(value_col)] = "#N/A"
    else:  # wrong_date
        f["date"] = "1999-01-01"
    return f, f"invalid:{kind}"


def build_inbox_files(raw: dict[str, pd.DataFrame], history_end: pd.Timestamp, end: pd.Timestamp,
                      cfg: dict) -> tuple[dict[str, pd.DataFrame], list[InboxFile]]:
    sources = cfg["sources"]
    an = cfg.get("anomalies", {})
    enabled = bool(an.get("enabled", True))
    seed = int(cfg.get("seed", 0))
    history: dict[str, pd.DataFrame] = {"stores": raw["stores"]}
    files: list[InboxFile] = []

    for source, scfg in sources.items():
        df = raw[source]
        gran = scfg.get("granularity", "day")
        lag = pd.Timedelta(days=int(scfg["lag_days"]))
        dd = _data_date(df, gran)
        delivered = dd + lag
        valid_date = df["_d"].notna()
        hist_mask = valid_date & (delivered < history_end)
        history[source] = df.loc[hist_mask].drop(columns="_d")
        inbox_mask = valid_date & (delivered >= history_end) & (delivered <= end)
        if (~valid_date).any():
            log.warning("%s: %d filas con fecha ilegible no se pueden ubicar en la bandeja",
                        source, int((~valid_date).sum()))
        sub = df.loc[inbox_mask].copy()
        sub["_dd"] = dd[inbox_mask]
        for data_date, frame in sub.groupby("_dd", sort=True):
            frame = frame.drop(columns=["_d", "_dd"]).reset_index(drop=True)
            base_delivery = data_date + lag
            rng = _rng(seed, source, f"{data_date:%Y-%m-%d}")
            u = rng.random()
            rates = [
                ("partial", an.get("partial_delivery_rate", 0) if source == "sales" else 0),
                ("correction", an.get("correction_rate", 0) if source == "sales" else 0),
                ("invalid", an.get("invalid_file_rate", 0) if source in ("sales", "transactions", "oil") else 0),
                ("late", an.get("late_delivery_rate", 0) if source in ("sales", "transactions", "oil") else 0),
            ]
            kind, acc = None, 0.0
            if enabled:
                for name, r in rates:
                    acc += float(r)
                    if u < acc:
                        kind = name
                        break
            v1 = frame
            if enabled and source in ("sales", "transactions") and rng.random() < an.get("duplicate_rows_rate", 0):
                dup = frame.sample(n=min(5, len(frame)), random_state=int(rng.integers(1 << 30)))
                v1 = pd.concat([frame, dup], ignore_index=True)

            if kind == "partial":
                stores = sorted(frame["store_nbr"].unique())
                k = max(1, int(len(stores) * an.get("partial_missing_store_frac", 0.3)))
                late_stores = set(rng.choice(stores, size=k, replace=False))
                in_late = frame["store_nbr"].isin(late_stores)
                files.append(InboxFile(source, data_date, 1, base_delivery, v1[~v1["store_nbr"].isin(late_stores)],
                                       "partial:part1"))
                files.append(InboxFile(source, data_date, 2, base_delivery + pd.Timedelta(days=1),
                                       frame[in_late], "partial:part2"))
            elif kind == "correction":
                n = max(1, int(len(frame) * an.get("correction_row_frac", 0.02)))
                idx = rng.choice(len(frame), size=n, replace=False)
                wrong = v1.copy()
                vals = pd.to_numeric(wrong["sales"], errors="coerce")
                factor = rng.choice([0.0, 10.0], size=n)
                vals.iloc[idx] = vals.iloc[idx] * factor
                wrong["sales"] = vals.astype(str)
                files.append(InboxFile(source, data_date, 1, base_delivery, wrong, "correction:wrong_values"))
                files.append(InboxFile(source, data_date, 2,
                                       base_delivery + pd.Timedelta(days=int(an.get("correction_lag_days", 5))),
                                       frame.iloc[idx], "correction:fix"))
            elif kind == "invalid":
                bad, note = _corrupt(v1, source, rng)
                files.append(InboxFile(source, data_date, 1, base_delivery, bad, note))
                files.append(InboxFile(source, data_date, 2,
                                       base_delivery + pd.Timedelta(days=int(an.get("invalid_fix_lag_days", 2))),
                                       frame, "invalid:resend"))
            elif kind == "late":
                k = int(rng.integers(1, int(an.get("late_max_days", 3)) + 1))
                files.append(InboxFile(source, data_date, 1, base_delivery + pd.Timedelta(days=k), v1, f"late:{k}d"))
            else:
                files.append(InboxFile(source, data_date, 1, base_delivery, v1,
                                       "duplicates" if len(v1) != len(frame) else ""))
    # nada se entrega después del fin de la simulación
    files = [f for f in files if f.delivered <= end]
    return history, files


def _source_cfg_hash(cfg: dict) -> str:
    return hashlib.sha256(repr(sorted(cfg.items(), key=str)).encode()).hexdigest()[:16]


def build_source_system(settings: Settings, force: bool = False) -> dict:
    """Construye (idempotentemente) la carga histórica y la bandeja."""
    kaggle_dir = settings.path("kaggle_dir")
    inbox_dir = settings.path("inbox_dir")
    history_dir = settings.path("history_dir")
    cfg = settings["source_system"]

    checksums = {f: sha256_file(kaggle_dir / f) for f in KAGGLE_FILES if (kaggle_dir / f).exists()}
    manifest_path = inbox_dir / MANIFEST
    current = read_json(manifest_path)
    if (not force and current and current.get("kaggle_checksums") == checksums
            and current.get("config_hash") == _source_cfg_hash(cfg)):
        log.info("Sistema de origen ya construido y vigente (%s); no se regenera.", manifest_path)
        return current

    external = current is None and inbox_dir.exists() and any(inbox_dir.iterdir())
    history_manifest = read_json(history_dir / MANIFEST)
    if external and not force and history_manifest and history_manifest.get("kaggle_checksums") == checksums:
        return history_manifest

    raw = read_kaggle(kaggle_dir)
    min_date, max_date = operating_range(raw["sales"]["_d"], raw["sales"]["store_nbr"])
    for name in raw:  # fechas fuera del rango operativo no se entregan (se registran arriba)
        if name != "stores":
            raw[name] = raw[name][raw[name]["_d"].isna() | (raw[name]["_d"] <= max_date + pd.Timedelta(days=62))]
    max_lag = max(int(s["lag_days"]) for s in cfg["sources"].values())

    if external:
        # La bandeja ya existe y no es del simulador (sistema externo): NO se toca. Sólo se construye la
        # carga histórica con lo entregado antes de la primera entrega de la bandeja.
        firsts = [pd.Timestamp(d.name.split("=", 1)[1]) for src in inbox_dir.iterdir() if src.is_dir()
                  for d in src.iterdir() if d.is_dir() and d.name.startswith("delivered=")]
        history_end = min(firsts)
        end = max(firsts)
        log.warning("Bandeja externa detectada en %s: sólo se construye la historia hasta %s.",
                    inbox_dir, history_end.date())
        history, files = build_inbox_files(raw, history_end, history_end - pd.Timedelta(days=1), cfg)
        if history_dir.exists():
            shutil.rmtree(history_dir)
        manifest_path = history_dir / MANIFEST
    else:
        history_end = resolve_history_end(min_date, max_date, cfg)
        end = max_date + pd.Timedelta(days=max(max_lag, 0))
        history, files = build_inbox_files(raw, history_end, end, cfg)
        for d in (inbox_dir, history_dir):
            if d.exists():
                shutil.rmtree(d)
    log.info("history_end=%s  max_data_date=%s  última entrega=%s", history_end.date(), max_date.date(), end.date())

    for source, df in history.items():
        write_parquet(df.reset_index(drop=True).astype(str), history_dir / f"{source}.parquet")
    for f in files:
        write_csv(f.frame, inbox_dir / f.relpath())

    anomalies = pd.DataFrame([{"source": f.source, "data_date": f"{f.data_date:%Y-%m-%d}", "version": f.version,
                               "delivered": f"{f.delivered:%Y-%m-%d}", "note": f.note, "rows": len(f.frame),
                               "file": f.relpath()} for f in files])
    reports = settings.path("reports_dir")
    if len(anomalies):
        write_csv(anomalies[anomalies["note"] != ""], reports / "source_system_anomalies.csv")

    manifest = {
        "history_end": f"{history_end:%Y-%m-%d}",
        "first_delivery": f"{history_end:%Y-%m-%d}",
        "last_delivery": f"{end:%Y-%m-%d}",
        "max_data_date": f"{max_date:%Y-%m-%d}",
        "n_files": len(files),
        "external_inbox": external,
        "kaggle_checksums": checksums,
        "config_hash": _source_cfg_hash(cfg),
    }
    write_json(manifest, manifest_path)
    return manifest
