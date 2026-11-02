"""Seguimiento del desempeño: pronósticos publicados vs. ventas reales conocidas.

Se recalcula completo en cada ejecución a partir de (a) los pronósticos
oficiales publicados y (b) las ventas *as-of* la fecha de ejecución. Por eso es
idempotente y refleja automáticamente las correcciones que vayan llegando.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Settings
from .io_utils import normalize_datetimes, write_csv, write_parquet, write_text
from .lake import load_asof
from .metrics import forecast_metrics
from .models import qcol
from .ordering import order_cost

log = logging.getLogger(__name__)


def _published(settings: Settings, until: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = settings.path("forecasts_dir")
    fcs, ords = [], []
    if base.exists():
        for d in sorted(base.glob("cutoff=*")):
            c = pd.Timestamp(d.name.split("=")[1])
            # estrictamente antes: el corte del mismo día aún no tiene ventas reales y
            # así el resultado no depende de si el DAG semanal ya terminó (idempotencia)
            if c < until and (d / "metadata.json").exists():
                fcs.append(normalize_datetimes(pd.read_parquet(d / "forecast.parquet")))
                ords.append(normalize_datetimes(pd.read_parquet(d / "orders.parquet")))
    if not fcs:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(fcs, ignore_index=True), pd.concat(ords, ignore_index=True)


def run_monitoring(settings: Settings, run_date) -> dict:
    run_date = pd.Timestamp(run_date).normalize()
    out = settings.path("monitoring_dir")
    levels = list(settings["forecast"]["quantiles"])
    fc, orders = _published(settings, run_date)
    if fc.empty:
        log.info("Aún no hay pronósticos publicados al %s; nada que evaluar (situación esperada).", run_date.date())
        return {"run_date": f"{run_date:%Y-%m-%d}", "cutoffs": 0}

    actual = load_asof(settings, "sales", run_date)[["date", "store_nbr", "family", "sales", "_delivered", "_version"]]
    actual["date"] = pd.to_datetime(actual["date"])
    actual = actual[actual["date"] < run_date]
    # día sin filas para ninguna serie pero con datos posteriores = cierre de cadena: venta real 0
    if len(actual):
        known = set(actual["date"].unique())
        span = pd.date_range(fc["date"].min() if "date" in fc else run_date, actual["date"].max(), freq="D")
        closed = [d for d in span if d not in known]
        if closed:
            keys = fc[pd.to_datetime(fc["date"]).isin(closed)][["date", "store_nbr", "family"]].drop_duplicates()
            actual = pd.concat([actual, keys.assign(sales=0.0, _delivered=pd.NaT, _version=0)], ignore_index=True)
    fc["date"] = pd.to_datetime(fc["date"])
    joined = fc.merge(actual.rename(columns={"sales": "y", "_delivered": "actual_delivered",
                                             "_version": "actual_version"}),
                      on=["date", "store_nbr", "family"], how="left")
    write_parquet(joined, out / "forecast_vs_actual.parquet")

    rows = []
    for (cutoff, model), g in joined.groupby(["cutoff", "model"]):
        due = g[g["date"] < run_date]
        m = forecast_metrics(g, levels)
        m.update(cutoff=f"{pd.Timestamp(cutoff):%Y-%m-%d}", model=model,
                 completeness=float(due["y"].notna().mean()) if len(due) else 0.0,
                 actual_rows_v2plus=int((g["actual_version"] > 1).sum()))
        # costo de negocio de pedidos con ventana de cobertura completamente observada
        o = orders[orders["cutoff"] == cutoff]
        if len(o):
            start, end = o["coverage_start"].iloc[0], o["coverage_end"].iloc[0]
            w = g[(g["date"] >= start) & (g["date"] <= end)]
            dem = w.groupby(["store_nbr", "family"])["y"].agg(["sum", "count"]).reset_index()
            days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
            dem = dem[dem["count"] == days]
            mo = o.merge(dem, on=["store_nbr", "family"])
            if len(mo):
                cu, co = mo["underage_cost"].to_numpy(), mo["overage_cost"].to_numpy()
                m["order_cost"] = float(order_cost(mo["order_qty"].to_numpy(), mo["sum"].to_numpy(), cu, co).sum())
                m["naive_order_cost"] = float(order_cost(mo["naive_order_qty"].to_numpy(), mo["sum"].to_numpy(),
                                                         cu, co).sum())
                m["orders_evaluated"] = int(len(mo))
                m["fill_rate"] = float(np.minimum(mo["order_qty"], mo["sum"]).sum() / max(mo["sum"].sum(), 1e-9))
        rows.append(m)
    by_cutoff = pd.DataFrame(rows).sort_values("cutoff")
    write_csv(by_cutoff, out / "metrics_by_cutoff.csv")

    recent_cut = sorted(joined["cutoff"].unique())[-4:]
    recent = joined[joined["cutoff"].isin(recent_cut)]
    fam_rows = []
    for fam, g in recent.groupby("family"):
        m = forecast_metrics(g, levels)
        m["family"] = fam
        fam_rows.append(m)
    by_family = pd.DataFrame(fam_rows)
    write_csv(by_family, out / "metrics_by_family_last4.csv")

    mon = settings["monitoring"]
    alerts = []
    evaluated = by_cutoff[by_cutoff["n"] > 0] if "n" in by_cutoff else by_cutoff.iloc[0:0]
    if len(evaluated):
        last = evaluated.iloc[-1]
        if last.get("wape", 0) > float(mon["wape_alert_threshold"]):
            alerts.append(f"WAPE {last['wape']:.3f} del corte {last['cutoff']} supera el umbral")
        cov = last.get("coverage_80")
        lo, hi = mon["coverage_alert_band"]
        if cov is not None and pd.notna(cov) and not (lo <= cov <= hi):
            alerts.append(f"Cobertura del intervalo 80% = {cov:.2f} fuera de [{lo}, {hi}] en {last['cutoff']}")
        lv = list(settings["forecast"]["quantiles"])
        hi_q, lo_q = qcol(lv[-1]), qcol(lv[0])
        if pd.notna(last.get(f"cal_{hi_q}")) and last[f"cal_{hi_q}"] < lv[-1] - 0.07:
            alerts.append(f"Cuantil {hi_q} subestima: P(y<={hi_q}) = {last[f'cal_{hi_q}']:.2f} en {last['cutoff']}")
        if pd.notna(last.get(f"cal_{lo_q}")) and last[f"cal_{lo_q}"] > lv[0] + 0.07:
            alerts.append(f"Cuantil {lo_q} sobreestima: P(y<={lo_q}) = {last[f'cal_{lo_q}']:.2f} en {last['cutoff']}")
    with_cost = (by_cutoff[by_cutoff["order_cost"].notna()] if "order_cost" in by_cutoff
                 else by_cutoff.iloc[0:0])
    if len(with_cost):
        lc = with_cost.iloc[-1]
        if lc["order_cost"] > lc["naive_order_cost"]:
            alerts.append(f"El pedido del corte {lc['cutoff']} costó más que la política ingenua "
                          f"({lc['order_cost']:.0f} vs {lc['naive_order_cost']:.0f})")
    for a in alerts:
        log.warning("ALERTA: %s", a)
    _report(settings, run_date, by_cutoff, by_family, alerts)
    return {"run_date": f"{run_date:%Y-%m-%d}", "cutoffs": int(len(by_cutoff)), "alerts": alerts}


def _fmt(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(sin datos)_\n"
    df = df.copy()
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def _report(settings: Settings, run_date, by_cutoff, by_family, alerts) -> None:
    cols = [c for c in ["cutoff", "model", "n", "completeness", "wape", "bias", "pinball", "coverage_80",
                        "cal_q10", "cal_q50", "cal_q90", "order_cost", "naive_order_cost", "fill_rate",
                        "actual_rows_v2plus"] if c in by_cutoff]
    txt = [f"# Seguimiento del desempeño — al {run_date:%Y-%m-%d}\n",
           "Pronósticos oficiales publicados vs. ventas reales conocidas a la fecha "
           "(incluye correcciones recibidas).\n",
           "## Alertas\n", ("\n".join(f"- {a}" for a in alerts) or "- Ninguna") + "\n",
           "## Por fecha de corte\n", _fmt(by_cutoff[cols]),
           "## Por familia (últimos 4 cortes)\n",
           _fmt(by_family[[c for c in ["family", "n", "wape", "bias", "coverage_80"] if c in by_family]])]
    write_text("\n".join(txt), settings.path("reports_dir") / "monitoring_report.md")
