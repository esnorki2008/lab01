"""Pasos de alto nivel que los DAGs (y la CLI) orquestan. Sin lógica de Airflow."""
from __future__ import annotations

import logging

import pandas as pd

from .config import Settings
from .forecasting import build_orders, generate_forecast, prepare_features, publish
from .io_utils import load_manifest
from .lake import ingest_until
from .monitoring import run_monitoring
from .training import train_and_register

log = logging.getLogger(__name__)


def first_cutoff(settings: Settings) -> pd.Timestamp | None:
    m = load_manifest(settings)
    if not m:
        return None
    d = pd.Timestamp(m["first_delivery"])
    wd = int(settings["forecast"]["cutoff_weekday"])
    return d + pd.Timedelta(days=(wd - d.weekday()) % 7)


def is_cutoff(settings: Settings, d: pd.Timestamp) -> bool:
    return pd.Timestamp(d).weekday() == int(settings["forecast"]["cutoff_weekday"])


def weekly(settings: Settings, cutoff, restate: bool = False) -> dict:
    cutoff = pd.Timestamp(cutoff)
    out = {"prepare": prepare_features(settings, cutoff, restate)}
    if not restate:
        out["train"] = train_and_register(settings, cutoff, first_cutoff=first_cutoff(settings))
    out["forecast"] = generate_forecast(settings, cutoff, restate)
    out["orders"] = build_orders(settings, cutoff, restate)
    out["publish"] = publish(settings, cutoff, restate)
    return out


def simulate(settings: Settings, start, end, monitor_every_day: bool = False) -> None:
    """Corre localmente la misma secuencia que Airflow, día por día."""
    for d in pd.date_range(start, end, freq="D"):
        ingest_until(settings, d)
        if is_cutoff(settings, d):
            weekly(settings, d)
        if monitor_every_day or is_cutoff(settings, d) or d == pd.Timestamp(end):
            run_monitoring(settings, d)
