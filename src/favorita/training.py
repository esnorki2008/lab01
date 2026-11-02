"""Validación temporal, criterio champion/challenger y entrenamiento del modelo vigente."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import Settings
from .features import Panel, build_panel, cutoff_features, training_frame
from .io_utils import code_version, read_json, write_csv, write_json
from .lake import build_snapshot
from .metrics import business_cost, forecast_metrics
from .models import TRAINABLE, make_model, save_model

log = logging.getLogger(__name__)

PANEL_HISTORY_DAYS = 380 + 112  # año anterior + ventanas largas


def _panel_for(settings: Settings, snap, asof: pd.Timestamp) -> Panel:
    f = settings["forecast"]
    start = asof - pd.Timedelta(days=int(f["train_window_days"]) + PANEL_HISTORY_DAYS + 7 * 8)
    end = asof + pd.Timedelta(days=int(f["horizon_days"]) - 1)
    return build_panel(snap, end=end, start=start)


def fold_cutoffs(settings: Settings, asof: pd.Timestamp) -> list[pd.Timestamp]:
    h = int(settings["forecast"]["horizon_days"])
    k = int(settings["model_selection"]["backtest_folds"])
    last = asof - pd.Timedelta(days=h)
    # alinear al día de corte
    wd = int(settings["forecast"]["cutoff_weekday"])
    last -= pd.Timedelta(days=(last.weekday() - wd) % 7)
    return sorted(last - pd.Timedelta(days=7 * i) for i in range(k))


def _train_rows(settings: Settings, panel: Panel, fold: pd.Timestamp) -> pd.DataFrame:
    f = settings["forecast"]
    tx_lag = int(settings["source_system"]["sources"]["transactions"]["lag_days"])
    return training_frame(panel, last_cutoff=fold - pd.Timedelta(days=int(f["train_cutoff_stride_days"])),
                          known_until=fold, horizon=int(f["horizon_days"]),
                          window_days=int(f["train_window_days"]), stride=int(f["train_cutoff_stride_days"]),
                          tx_lag=tx_lag)


def _fold_test(settings: Settings, panel: Panel, fold: pd.Timestamp, h: int, tx_lag: int) -> pd.DataFrame:
    """Filas de evaluación de un fold con los features que REALMENTE se tenían en el fold.

    Los features se construyen con la vista as-of del fold (con sus atrasos, entregas parciales
    y valores aún sin corregir), igual que en producción; el valor real ``y`` sale de la vista
    más reciente. Así el backtest mide el desempeño en las condiciones en que se opera.
    """
    keys = ["store_nbr", "family", "date"]
    truth = cutoff_features(panel, fold, h, tx_lag, with_target=True)[keys + ["y"]]
    snap_fold = build_snapshot(settings, fold)
    fold_panel = build_panel(snap_fold, end=fold + pd.Timedelta(days=h - 1),
                             start=fold - pd.Timedelta(days=PANEL_HISTORY_DAYS + 56))
    feats = cutoff_features(fold_panel, fold, h, tx_lag)
    return feats.merge(truth, on=keys, how="inner").reset_index(drop=True)


def backtest(settings: Settings, asof: pd.Timestamp, candidates: list[str] | None = None,
             snap=None) -> pd.DataFrame:
    """Rolling-origin: mismos cortes y mismos datos para todos los modelos."""
    asof = pd.Timestamp(asof)
    snap = snap or build_snapshot(settings, asof)
    panel = _panel_for(settings, snap, asof)
    candidates = candidates or list(settings["model_selection"]["candidates"])
    levels = list(settings["forecast"]["quantiles"])
    h = int(settings["forecast"]["horizon_days"])
    tx_lag = int(settings["source_system"]["sources"]["transactions"]["lag_days"])
    rows = []
    for fold in fold_cutoffs(settings, asof):
        test = _fold_test(settings, panel, fold, h, tx_lag)
        naive_daily = test.drop_duplicates(["store_nbr", "family"]).set_index(["store_nbr", "family"])["mean_7"]
        train = None
        for name in candidates:
            model = make_model(name, settings)
            if name in TRAINABLE:
                if train is None:
                    train = _train_rows(settings, panel, fold)
                if len(train) < 100:
                    log.warning("fold %s: datos insuficientes para %s", fold.date(), name)
                    continue
                model.fit(train)
            pred = pd.concat([test, model.predict(test)], axis=1)
            met = forecast_metrics(pred, levels)
            met.update(business_cost(pred, settings, naive_daily))
            met.update(model=name, fold=f"{fold:%Y-%m-%d}")
            rows.append(met)
            log.info("backtest %s %s: wape=%.3f cost=%.1f", fold.date(), name, met.get("wape", float("nan")),
                     met.get("business_cost", float("nan")))
    return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    if bt.empty:
        return bt
    g = bt.groupby("model")
    return pd.DataFrame({
        "wape": g["wape"].mean(), "bias": g["bias"].mean(), "rmsle": g["rmsle"].mean(),
        "pinball": g["pinball"].mean(), "coverage_80": g["coverage_80"].mean(),
        "business_cost": g["business_cost"].sum(), "naive_cost": g["naive_cost"].sum(),
        "fill_rate": g["fill_rate"].mean(), "folds": g.size(),
    }).reset_index()


def _fold_win_rate(bt: pd.DataFrame | None, challenger: str, champion: str) -> float:
    if bt is None or bt.empty:
        return 1.0
    piv = bt.pivot_table(index="fold", columns="model", values="business_cost")
    if challenger not in piv or champion not in piv:
        return 0.0
    both = piv[[challenger, champion]].dropna()
    return float((both[challenger] < both[champion]).mean()) if len(both) else 0.0


def decide(summary: pd.DataFrame, champion: str, settings: Settings, bt: pd.DataFrame | None = None) -> dict:
    """Criterio explícito de reemplazo (ver params.yaml: model_selection)."""
    ms = settings["model_selection"]
    s = summary.set_index("model")
    if champion not in s.index:
        # el champion no pudo evaluarse: cualquier candidato válido lo reemplaza
        best = s["business_cost"].idxmin()
        return {"champion": best, "previous": champion, "promoted": best != champion,
                "reason": "champion sin evaluación"}
    ch = s.loc[champion]
    best_name, best_reason = champion, "ningún challenger cumple el criterio"
    best_cost = ch["business_cost"]
    for name, r in s.drop(index=champion).iterrows():
        checks = {
            "cost": r["business_cost"] <= ch["business_cost"] * (1 - float(ms["min_cost_improvement"])),
            "wape": r["wape"] <= ch["wape"] * (1 + float(ms["max_wape_degradation"])),
            "coverage": (pd.notna(r["coverage_80"])
                         and abs(r["coverage_80"] - 0.8) <= float(ms["coverage_tolerance"])),
            # debe ganar en costo en la mayoría de los folds, no sólo en el total
            "folds": _fold_win_rate(bt, name, champion) >= float(ms.get("min_fold_win_rate", 0.5)),
        }
        if all(checks.values()) and r["business_cost"] < best_cost:
            best_name, best_cost = name, r["business_cost"]
            best_reason = (f"{name}: costo {r['business_cost']:.0f} vs {ch['business_cost']:.0f}, "
                           f"WAPE {r['wape']:.3f} vs {ch['wape']:.3f}, cobertura {r['coverage_80']:.2f}")
        elif not all(checks.values()):
            log.info("challenger %s no cumple: %s", name, {k: bool(v) for k, v in checks.items()})
    return {"champion": best_name, "previous": champion, "promoted": best_name != champion, "reason": best_reason}


# ----------------------------------------------------------------------------
# Registro de modelos
# ----------------------------------------------------------------------------
def registry_path(settings: Settings) -> Path:
    return settings.path("models_dir") / "registry.json"


def load_registry(settings: Settings) -> dict:
    return read_json(registry_path(settings), {"entries": {}})


def champion_entry(settings: Settings, cutoff: pd.Timestamp) -> dict | None:
    """Última decisión de entrenamiento con fecha <= cutoff (lo vigente en ese momento)."""
    entries = load_registry(settings)["entries"]
    keys = sorted(k for k in entries if pd.Timestamp(k) <= pd.Timestamp(cutoff))
    return entries[keys[-1]] if keys else None


def needs_training(settings: Settings, cutoff: pd.Timestamp, first_cutoff: pd.Timestamp | None) -> bool:
    cutoff = pd.Timestamp(cutoff)
    entries = load_registry(settings)["entries"]
    if f"{cutoff:%Y-%m-%d}" in entries:
        return False  # ya entrenado para este corte (idempotencia)
    if champion_entry(settings, cutoff) is None:
        return True
    every = int(settings["forecast"]["retrain_every_weeks"])
    ref = pd.Timestamp(first_cutoff) if first_cutoff is not None else pd.Timestamp("1970-01-05")
    return ((cutoff - ref).days // 7) % every == 0


def train_and_register(settings: Settings, cutoff: str | pd.Timestamp, force: bool = False,
                       first_cutoff: pd.Timestamp | None = None) -> dict:
    cutoff = pd.Timestamp(cutoff).normalize()
    key = f"{cutoff:%Y-%m-%d}"
    if not force and not needs_training(settings, cutoff, first_cutoff):
        entry = champion_entry(settings, cutoff)
        log.info("No corresponde reentrenar en %s; vigente: %s", key, entry and entry.get("champion"))
        return {"trained": False, "entry": entry}

    snap = build_snapshot(settings, cutoff)
    prev = champion_entry(settings, cutoff - pd.Timedelta(days=1))
    current = prev["champion"] if prev else settings["model_selection"]["initial_champion"]
    bt = backtest(settings, cutoff, snap=snap)
    summary = summarize_backtest(bt)
    decision = decide(summary, current, settings, bt) if len(summary) else {
        "champion": current, "previous": current, "promoted": False, "reason": "sin datos para backtest"}

    model_path = None
    name = decision["champion"]
    model = make_model(name, settings)
    if name in TRAINABLE:
        panel = _panel_for(settings, snap, cutoff)
        f = settings["forecast"]
        train = training_frame(panel, last_cutoff=cutoff - pd.Timedelta(days=int(f["train_cutoff_stride_days"])),
                               known_until=cutoff, horizon=int(f["horizon_days"]),
                               window_days=int(f["train_window_days"]),
                               stride=int(f["train_cutoff_stride_days"]),
                               tx_lag=int(settings["source_system"]["sources"]["transactions"]["lag_days"]))
        model.fit(train)
        rel = Path(name) / f"trained={key}" / "model.joblib"
        save_model(model, settings.path("models_dir") / rel)
        model_path = rel.as_posix()
    write_csv(bt, settings.path("models_dir") / "backtests" / f"backtest_{key}.csv")
    entry = {**decision, "trained_cutoff": key, "model_path": model_path,
             "backtest": summary.round(5).to_dict(orient="records"),
             "code_version": code_version(settings.root), "config_hash": settings.config_hash}
    reg = load_registry(settings)
    reg["entries"][key] = entry
    write_json(reg, registry_path(settings))
    log.info("Decisión %s: %s", key, decision)
    return {"trained": True, "entry": entry}
