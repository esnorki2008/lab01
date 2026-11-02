"""Datos sintéticos con el MISMO formato que Kaggle (para pruebas y CI)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = ["BEVERAGES", "BREAD/BAKERY", "DAIRY", "GROCERY I", "PRODUCE", "AUTOMOTIVE"]


def make_sample_kaggle(n_stores: int = 4, days: int = 420, end: str = "2017-08-15",
                       seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end, periods=days, freq="D")
    stores = pd.DataFrame({
        "store_nbr": range(1, n_stores + 1),
        "city": ["Quito", "Quito", "Guayaquil", "Cuenca", "Ambato", "Loja"][:n_stores],
        "state": ["Pichincha", "Pichincha", "Guayas", "Azuay", "Tungurahua", "Loja"][:n_stores],
        "type": list("ABCDE")[:n_stores],
        "cluster": [13, 13, 8, 4, 2, 1][:n_stores],
    })
    rows = []
    dow_effect = np.array([1.0, 0.9, 0.9, 0.95, 1.1, 1.4, 1.3])
    for s in stores["store_nbr"]:
        for fam in FAMILIES:
            level = 0 if fam == "AUTOMOTIVE" and s == 1 else rng.uniform(5, 200)
            promo = (rng.random(days) < 0.15).astype(int) * rng.integers(1, 10, days)
            season = 1 + 0.1 * np.sin(np.arange(days) / 58)
            mu = level * dow_effect[dates.dayofweek] * (1 + 0.3 * (promo > 0)) * season
            sales = rng.poisson(np.maximum(mu, 0)).astype(float)
            rows.append(pd.DataFrame({"date": dates, "store_nbr": s, "family": fam, "sales": sales,
                                      "onpromotion": promo}))
    train = pd.concat(rows, ignore_index=True).sort_values(["date", "store_nbr", "family"])
    train.insert(0, "id", range(len(train)))
    train["date"] = train["date"].dt.strftime("%Y-%m-%d")
    tx = train.groupby(["date", "store_nbr"])["sales"].sum().reset_index()
    tx["transactions"] = (tx["sales"] * 1.7 + 100).round().astype(int)
    tx = tx.drop(columns="sales")
    oil_dates = pd.date_range(dates[0], pd.Timestamp(end) + pd.Timedelta(days=16), freq="B")
    oil = pd.DataFrame({"date": oil_dates.strftime("%Y-%m-%d"),
                        "dcoilwtico": (50 + np.cumsum(rng.normal(0, 0.5, len(oil_dates)))).round(2)})
    oil.loc[oil.index[::37], "dcoilwtico"] = np.nan
    hol = []
    for y in sorted(set(dates.year)) + [dates[-1].year + 1]:
        hol.append((f"{y}-01-01", "Holiday", "National", "Ecuador", "Primer dia del ano", False))
        hol.append((f"{y}-05-24", "Holiday", "National", "Ecuador", "Batalla de Pichincha", True))
        hol.append((f"{y}-05-26", "Transfer", "National", "Ecuador", "Traslado Batalla de Pichincha", False))
        hol.append((f"{y}-12-06", "Holiday", "Local", "Quito", "Fundacion de Quito", False))
        hol.append((f"{y}-11-03", "Holiday", "Local", "Cuenca", "Independencia de Cuenca", False))
        hol.append((f"{y}-12-25", "Holiday", "National", "Ecuador", "Navidad", False))
    holidays = pd.DataFrame(hol, columns=["date", "type", "locale", "locale_name", "description", "transferred"])
    return {"train": train, "stores": stores, "oil": oil, "transactions": tx, "holidays_events": holidays}


def write_sample_kaggle(out: Path, **kw) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for name, df in make_sample_kaggle(**kw).items():
        df.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    return out
