"""Modelos de pronóstico probabilístico (cuantiles).

* ``baseline``: cuantiles empíricos de las últimas 8 ventas del mismo día de
  la semana (naive estacional). No se entrena; sirve de piso de comparación.
* ``lgbm_quantile``: LightGBM global (todas las series) con objetivo
  *quantile*, un modelo por cuantil, entrenado sobre log1p(ventas). Como los
  cuantiles son invariantes a transformaciones monótonas, expm1 del cuantil
  en log es el cuantil en unidades.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from .features import DOW_LAGS, FEATURES

QCOLS_PREFIX = "q"


def qcol(q: float) -> str:
    return f"{QCOLS_PREFIX}{int(round(q * 100)):02d}"


class ForecastModel(Protocol):
    name: str
    quantiles: list[float]

    def fit(self, train: pd.DataFrame) -> "ForecastModel": ...
    def predict(self, feats: pd.DataFrame) -> pd.DataFrame: ...


def _sort_quantiles(pred: np.ndarray) -> np.ndarray:
    return np.sort(np.clip(pred, 0, None), axis=1)


class BaselineModel:
    name = "baseline"

    def __init__(self, quantiles: list[float], **_: object) -> None:
        self.quantiles = list(quantiles)

    def fit(self, train: pd.DataFrame) -> "BaselineModel":
        return self

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        hist = feats[[f"dow_lag_{i + 1}" for i in range(DOW_LAGS)]].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pred = np.nanquantile(hist, self.quantiles, axis=1).T
        fallback = feats["mean_28"].to_numpy(dtype=float)
        fallback = np.where(np.isnan(fallback), 0.0, fallback)
        pred = np.where(np.isnan(pred), fallback[:, None], pred)
        return pd.DataFrame(_sort_quantiles(pred), columns=[qcol(q) for q in self.quantiles], index=feats.index)


class LGBMQuantileModel:
    name = "lgbm_quantile"

    def __init__(self, quantiles: list[float], **params: object) -> None:
        self.quantiles = list(quantiles)
        self.params = dict(params)
        self.boosters: dict[float, object] = {}
        self.features = list(FEATURES)

    def fit(self, train: pd.DataFrame) -> "LGBMQuantileModel":
        import lightgbm as lgb

        X = train[self.features].astype(float)
        y = np.log1p(train["y"].clip(lower=0).to_numpy(dtype=float))
        p = dict(self.params)
        n_estimators = int(p.pop("n_estimators", 300))
        threads = int(p.pop("num_threads", 4))
        seed = int(p.pop("seed", 7))
        for q in self.quantiles:
            params = {"objective": "quantile", "alpha": q, "verbosity": -1, "num_threads": threads,
                      "seed": seed, "deterministic": True, "force_row_wise": True, **p}
            ds = lgb.Dataset(X, y, categorical_feature=["family_code", "store_nbr", "store_type_code",
                                                        "city_code"], free_raw_data=True)
            self.boosters[q] = lgb.train(params, ds, num_boost_round=n_estimators)
        return self

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        X = feats[self.features].astype(float)
        pred = np.column_stack([np.expm1(self.boosters[q].predict(X)) for q in self.quantiles])
        # una serie sin ventas en 8 semanas (tienda cerrada / producto descontinuado) => 0
        dead = (feats["mean_56"].fillna(0) == 0) & (feats["known_days_7"] > 0)
        # sin ninguna venta en la historia disponible (nunca vendió) => 0
        dead |= feats[["mean_112", "dow_mean_8"]].isna().all(axis=1)
        pred[dead.to_numpy()] = 0.0
        return pd.DataFrame(_sort_quantiles(pred), columns=[qcol(q) for q in self.quantiles], index=feats.index)


MODEL_CLASSES = {"baseline": BaselineModel, "lgbm_quantile": LGBMQuantileModel}
TRAINABLE = {"lgbm_quantile"}


def make_model(name: str, settings) -> ForecastModel:
    quantiles = settings["forecast"]["quantiles"]
    params = (settings["models"] or {}).get(name, {}) or {}
    return MODEL_CLASSES[name](quantiles, **params)


def save_model(model: ForecastModel, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(model, tmp)
    tmp.replace(path)
    return path


def load_model(path: Path) -> ForecastModel:
    return joblib.load(path)
