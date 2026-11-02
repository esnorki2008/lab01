"""Pedido sugerido a partir del pronóstico probabilístico y de los costos.

Problema del vendedor de periódicos (newsvendor): con costo unitario de
quedarse sin producto ``cu`` y de merma/sobrestock ``co``, la cantidad óptima
es el cuantil ``cu / (cu + co)`` de la demanda del periodo a cubrir.

Ventana de cobertura: el pedido emitido en el corte T llega en T + lead_time y
debe cubrir ``review_period`` días. La cantidad es la suma de los cuantiles
diarios en la ventana (exacto bajo dependencia comonotónica entre días; ver
DECISIONES.md). Los cuantiles intermedios se interpolan linealmente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import qcol


def critical_ratio(cu: float, co: float) -> float:
    return cu / (cu + co)


def interpolate_quantile(qvals: np.ndarray, levels: list[float], alpha: np.ndarray) -> np.ndarray:
    """qvals: (n, k) cuantiles ordenados; alpha: (n,). Se acota al rango estimado."""
    levels_a = np.asarray(levels, dtype=float)
    a = np.clip(alpha, levels_a[0], levels_a[-1])
    idx = np.clip(np.searchsorted(levels_a, a, side="right") - 1, 0, len(levels_a) - 2)
    lo, hi = levels_a[idx], levels_a[idx + 1]
    w = (a - lo) / (hi - lo)
    rows = np.arange(len(a))
    return qvals[rows, idx] * (1 - w) + qvals[rows, idx + 1] * w


def coverage_window(cutoff: pd.Timestamp, settings) -> tuple[pd.Timestamp, pd.Timestamp]:
    b = settings["business"]
    start = cutoff + pd.Timedelta(days=int(b["lead_time_days"]))
    end = start + pd.Timedelta(days=int(b["review_period_days"]) - 1)
    horizon_end = cutoff + pd.Timedelta(days=int(settings["forecast"]["horizon_days"]) - 1)
    if end > horizon_end:
        raise ValueError("La ventana de cobertura excede el horizonte de pronóstico; revise params.yaml")
    return start, end


def suggest_orders(forecast: pd.DataFrame, settings) -> pd.DataFrame:
    levels = list(settings["forecast"]["quantiles"])
    cutoff = pd.Timestamp(forecast["cutoff"].iloc[0])
    start, end = coverage_window(cutoff, settings)
    win = forecast[(forecast["date"] >= start) & (forecast["date"] <= end)].copy()
    fams = win["family"].unique()
    costs = {f: settings.family_costs(f) for f in fams}
    win["underage_cost"] = win["family"].map(lambda f: costs[f][0])
    win["overage_cost"] = win["family"].map(lambda f: costs[f][1])
    win["service_level"] = win["underage_cost"] / (win["underage_cost"] + win["overage_cost"])
    qv = win[[qcol(q) for q in levels]].to_numpy(dtype=float)
    win["q_alpha"] = interpolate_quantile(qv, levels, win["service_level"].to_numpy())
    win["q_median"] = interpolate_quantile(qv, levels, np.full(len(win), 0.5))
    agg = (win.groupby(["store_nbr", "family"], as_index=False)
              .agg(expected_demand=("q_median", "sum"), order_qty=("q_alpha", "sum"),
                   service_level=("service_level", "first"), underage_cost=("underage_cost", "first"),
                   overage_cost=("overage_cost", "first")))
    agg["order_qty"] = agg["order_qty"].round(2)
    agg["expected_demand"] = agg["expected_demand"].round(2)
    agg.insert(0, "cutoff", cutoff)
    agg.insert(1, "coverage_start", start)
    agg.insert(2, "coverage_end", end)
    return agg


def order_cost(order_qty: np.ndarray, demand: np.ndarray, cu: np.ndarray, co: np.ndarray) -> np.ndarray:
    return cu * np.maximum(demand - order_qty, 0) + co * np.maximum(order_qty - demand, 0)
