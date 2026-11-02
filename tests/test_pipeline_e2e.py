"""Prueba de punta a punta: la misma secuencia que ejecutan los DAGs, sobre datos sintéticos."""
import shutil

import pandas as pd
import pytest

from favorita.forecasting import DataQualityError, prepare_features, reproduce
from favorita.io_utils import read_json
from favorita.monitoring import run_monitoring
from favorita.pipeline import weekly
from favorita.training import load_registry


def _cutoffs(s):
    return sorted(p.name.split("=")[1] for p in s.path("forecasts_dir").glob("cutoff=*"))


def test_forecasts_and_orders_published_every_week(simulated):
    s, m = simulated
    cuts = _cutoffs(s)
    assert len(cuts) >= 6
    assert all(pd.Timestamp(c).weekday() == 0 for c in cuts)
    d = s.path("forecasts_dir") / f"cutoff={cuts[-1]}"
    fc = pd.read_parquet(d / "forecast.parquet")
    assert {"q10", "q50", "q90"} <= set(fc.columns) and fc["h"].nunique() == 14
    orders = pd.read_csv(d / "orders.csv")
    assert (orders["order_qty"] >= 0).all() and len(orders) == fc[["store_nbr", "family"]].drop_duplicates().shape[0]
    meta = read_json(d / "metadata.json")
    assert meta["kind"] == "official" and meta["n_input_files"] > 0


def test_registry_has_champion_decisions(simulated):
    s, _ = simulated
    entries = load_registry(s)["entries"]
    assert entries
    for e in entries.values():
        assert e["champion"] in ("baseline", "lgbm_quantile") and e["backtest"]


def test_published_forecast_is_reproducible(simulated, tmp_path):
    s, _ = simulated
    for c in _cutoffs(s)[:2]:
        r = reproduce(s, c, tmp_path / c)
        assert r["identical"], r


def test_rerun_is_idempotent_and_does_not_overwrite_published(simulated):
    s, _ = simulated
    c = _cutoffs(s)[1]
    before = read_json(s.path("forecasts_dir") / f"cutoff={c}" / "metadata.json")
    r = weekly(s, c)
    assert r["publish"]["status"] == "unchanged"
    assert read_json(s.path("forecasts_dir") / f"cutoff={c}" / "metadata.json") == before


def test_restatement_uses_corrections_but_keeps_official(simulated):
    s, _ = simulated
    c = _cutoffs(s)[0]
    official = read_json(s.path("forecasts_dir") / f"cutoff={c}" / "metadata.json")
    r = weekly(s, c, restate=True)
    assert "restated" in r["publish"]["path"]
    assert read_json(s.path("forecasts_dir") / f"cutoff={c}" / "metadata.json") == official


def test_monitoring_compares_with_actuals(simulated):
    s, m = simulated
    res = run_monitoring(s, m["last_delivery"])
    assert res["cutoffs"] >= 5
    mc = pd.read_csv(s.path("monitoring_dir") / "metrics_by_cutoff.csv")
    done = mc[mc["completeness"] == 1.0]
    assert len(done) and done["wape"].between(0, 1).all()
    assert "order_cost" in mc and mc["order_cost"].notna().any()
    assert (s.path("reports_dir") / "monitoring_report.md").exists()
    # idempotente
    first = (s.path("monitoring_dir") / "metrics_by_cutoff.csv").read_bytes()
    run_monitoring(s, m["last_delivery"])
    assert (s.path("monitoring_dir") / "metrics_by_cutoff.csv").read_bytes() == first


def test_stale_data_stops_forecast(simulated, tmp_path):
    s, m = simulated
    # un corte muy posterior al último dato disponible es un incidente: no se publica
    late = pd.Timestamp(m["last_delivery"]) + pd.Timedelta(days=21)
    late += pd.Timedelta(days=(7 - late.weekday()) % 7)
    with pytest.raises(DataQualityError):
        prepare_features(s, late, root=tmp_path)
    shutil.rmtree(tmp_path, ignore_errors=True)
