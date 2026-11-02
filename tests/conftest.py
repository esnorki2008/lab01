"""Fixtures: proyecto aislado en un directorio temporal con datos sintéticos de formato Kaggle."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import favorita  # noqa: F401  (orden de carga de DLLs en Windows)
from favorita.config import load_settings
from favorita.sample_data import write_sample_kaggle

REPO = Path(__file__).resolve().parents[1]

FAST = {
    "source_system": {"inbox_days": 49},
    "forecast": {"train_window_days": 35, "retrain_every_weeks": 2},
    "models": {"lgbm_quantile": {"n_estimators": 15, "num_leaves": 7, "min_child_samples": 5, "num_threads": 1}},
    "model_selection": {"backtest_folds": 2},
    "versioning": {"dvc_auto_snapshot": False},
}


def make_project(tmp: Path, overrides: dict | None = None, **sample_kw) -> object:
    (tmp / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "config" / "params.yaml", tmp / "config" / "params.yaml")
    write_sample_kaggle(tmp / "data" / "store-sales-time-series-forecasting", **sample_kw)
    from favorita.config import deep_merge
    return load_settings(root=tmp, overrides=deep_merge(FAST, overrides or {}))


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path, days=300)


@pytest.fixture(scope="session")
def simulated(tmp_path_factory):
    """Proyecto con el flujo completo ya ejecutado sobre todo el periodo de la bandeja."""
    from favorita.pipeline import simulate
    from favorita.source_system import build_source_system

    tmp = tmp_path_factory.mktemp("sim")
    s = make_project(tmp, days=300)
    m = build_source_system(s)
    simulate(s, m["first_delivery"], m["last_delivery"])
    return s, m
