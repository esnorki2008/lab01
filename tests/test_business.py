import numpy as np
import pandas as pd
import pytest

from favorita.config import deep_merge
from favorita.metrics import interval_coverage, scaled_pinball, wape
from favorita.ordering import critical_ratio, interpolate_quantile, order_cost, suggest_orders
from favorita.training import decide


def test_critical_ratio_and_interpolation():
    assert critical_ratio(1, 0.25) == pytest.approx(0.8)
    q = np.array([[10.0, 20.0, 30.0]])
    assert interpolate_quantile(q, [0.1, 0.5, 0.9], np.array([0.7]))[0] == pytest.approx(25.0)
    assert interpolate_quantile(q, [0.1, 0.5, 0.9], np.array([0.99]))[0] == pytest.approx(30.0)


def _forecast(cutoff="2017-03-06"):
    c = pd.Timestamp(cutoff)
    rows = []
    for fam in ("DAIRY", "GROCERY I"):
        for h in range(14):
            rows.append({"cutoff": c, "date": c + pd.Timedelta(days=h), "h": h, "store_nbr": 1, "family": fam,
                         "q10": 5.0, "q25": 8.0, "q50": 10.0, "q75": 12.0, "q90": 15.0})
    return pd.DataFrame(rows)


def test_costs_in_config_change_the_order_without_code_changes(project):
    o1 = suggest_orders(_forecast(), project).set_index("family")
    assert o1.loc["DAIRY", "service_level"] < o1.loc["GROCERY I", "service_level"]  # perecible pide menos
    assert o1.loc["DAIRY", "order_qty"] < o1.loc["GROCERY I", "order_qty"]
    project.raw = deep_merge(project.raw, {"business": {"family_overrides": {
        "GROCERY I": {"underage_cost": 1.0, "overage_cost": 9.0}}}})
    o2 = suggest_orders(_forecast(), project).set_index("family")
    assert o2.loc["GROCERY I", "order_qty"] < o1.loc["GROCERY I", "order_qty"]
    # ventana: lead 2 días, 7 días de revisión
    assert (o1["coverage_end"] - o1["coverage_start"]).dt.days.iloc[0] == 6


def test_order_cost_asymmetric():
    c = order_cost(np.array([10.0, 10.0]), np.array([15.0, 5.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5]))
    assert c.tolist() == [5.0, 2.5]


def test_metrics():
    y = np.array([10.0, 0.0, 10.0])
    assert wape(y, y) == 0 and wape(y, np.zeros(3)) == 1
    q = np.column_stack([y - 1, y, y + 1])
    assert scaled_pinball(y, q, [0.1, 0.5, 0.9]) < scaled_pinball(y, q + 5, [0.1, 0.5, 0.9])
    assert interval_coverage(y, y - 1, y + 1) == 1.0


def _summary(cost_c, wape_c, cov_c=0.8):
    return pd.DataFrame([
        {"model": "baseline", "business_cost": 100.0, "wape": 0.20, "coverage_80": 0.7},
        {"model": "lgbm_quantile", "business_cost": cost_c, "wape": wape_c, "coverage_80": cov_c},
    ])


def test_champion_challenger_rule(project):
    assert decide(_summary(80, 0.19), "baseline", project)["champion"] == "lgbm_quantile"
    assert decide(_summary(99, 0.15), "baseline", project)["champion"] == "baseline"   # mejora < 3%
    assert decide(_summary(80, 0.30), "baseline", project)["champion"] == "baseline"   # WAPE empeora
    assert decide(_summary(80, 0.19, 0.4), "baseline", project)["champion"] == "baseline"  # mala cobertura
