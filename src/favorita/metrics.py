"""Métricas técnicas del pronóstico y métrica de negocio."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import qcol
from .ordering import interpolate_quantile, order_cost, suggest_orders


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    d = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - yhat)) / d) if d > 0 else float("nan")


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    d = np.sum(np.abs(y))
    return float(np.sum(yhat - y) / d) if d > 0 else float("nan")


def rmsle(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(np.clip(yhat, 0, None)) - np.log1p(np.clip(y, 0, None))) ** 2)))


def scaled_pinball(y: np.ndarray, qvals: np.ndarray, levels: list[float]) -> float:
    """Pérdida pinball promedio sobre cuantiles, escalada por la media de y."""
    losses = []
    for i, q in enumerate(levels):
        diff = y - qvals[:, i]
        losses.append(np.mean(np.maximum(q * diff, (q - 1) * diff)))
    m = np.mean(np.abs(y))
    return float(np.mean(losses) / m) if m > 0 else float("nan")


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi))) if len(y) else float("nan")


def forecast_metrics(df: pd.DataFrame, levels: list[float]) -> dict:
    """df con columna ``y`` (real) y columnas de cuantiles."""
    df = df[df["y"].notna()]
    if not len(df):
        return {"n": 0}
    y = df["y"].to_numpy(dtype=float)
    qv = df[[qcol(q) for q in levels]].to_numpy(dtype=float)
    med = interpolate_quantile(qv, levels, np.full(len(df), 0.5))
    out = {
        "n": int(len(df)),
        "wape": wape(y, med),
        "bias": bias(y, med),
        "rmsle": rmsle(y, med),
        "pinball": scaled_pinball(y, qv, levels),
        "coverage_80": interval_coverage(y, qv[:, 0], qv[:, -1]) if levels[0] <= 0.1 and levels[-1] >= 0.9 else None,
    }
    # calibración de cada cuantil: P(y <= q_tau) debería ser ~tau (la cobertura del intervalo puede
    # ocultar que un extremo esté mal calibrado)
    for i, q in enumerate(levels):
        out[f"cal_{qcol(q)}"] = float(np.mean(y <= qv[:, i]))
    return out


def business_cost(forecast_with_y: pd.DataFrame, settings, naive_daily: pd.Series | None = None) -> dict:
    """Costo de negocio del pedido derivado del pronóstico, evaluado con la demanda real.

    ``naive_daily``: demanda diaria promedio reciente por serie (índice store_nbr, family)
    para la política actual (pedir lo vendido la semana anterior).
    """
    orders = suggest_orders(forecast_with_y, settings)
    win = forecast_with_y[(forecast_with_y["date"] >= orders["coverage_start"].iloc[0])
                          & (forecast_with_y["date"] <= orders["coverage_end"].iloc[0])]
    demand = win.groupby(["store_nbr", "family"])["y"].agg(["sum", "count"]).reset_index()
    days = int(settings["business"]["review_period_days"])
    demand = demand[demand["count"] == days]   # sólo ventanas completamente observadas
    m = orders.merge(demand, on=["store_nbr", "family"], how="inner")
    if not len(m):
        return {"business_cost": float("nan"), "n_orders": 0}
    cu, co = m["underage_cost"].to_numpy(), m["overage_cost"].to_numpy()
    cost = order_cost(m["order_qty"].to_numpy(), m["sum"].to_numpy(), cu, co)
    out = {"business_cost": float(cost.sum()), "n_orders": int(len(m)),
           "demand": float(m["sum"].sum()),
           "fill_rate": float(np.minimum(m["order_qty"], m["sum"]).sum() / max(m["sum"].sum(), 1e-9)),
           "overstock_units": float(np.maximum(m["order_qty"] - m["sum"], 0).sum())}
    out["cost_per_unit_demand"] = out["business_cost"] / max(out["demand"], 1e-9)
    if naive_daily is not None:
        naive = m.set_index(["store_nbr", "family"]).index.map(lambda k: naive_daily.get(k, np.nan))
        naive_q = np.nan_to_num(np.asarray(naive, dtype=float)) * days
        out["naive_cost"] = float(order_cost(naive_q, m["sum"].to_numpy(), cu, co).sum())
    return out
