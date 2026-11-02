import numpy as np
import pandas as pd

from favorita.features import FEATURES, build_panel, cutoff_features
from favorita.lake import Snapshot
from favorita.models import BaselineModel, LGBMQuantileModel


def _snap(days=200):
    dates = pd.date_range("2017-01-01", periods=days)
    rows = []
    for s in (1, 2):
        for f in ("A", "B"):
            rows.append(pd.DataFrame({"date": dates, "store_nbr": s, "family": f,
                                      "sales": (np.arange(days) % 7 + 1) * (10 if f == "A" else 1.0)}))
    sales = pd.concat(rows)
    promo = sales[["date", "store_nbr", "family"]].assign(onpromotion=0)
    stores = pd.DataFrame({"store_nbr": [1, 2], "city": ["Quito", "Cuenca"], "state": ["P", "A"],
                           "type": ["A", "B"], "cluster": [1, 2]})
    tx = sales.groupby(["date", "store_nbr"], as_index=False)["sales"].sum().rename(columns={"sales": "transactions"})
    oil = pd.DataFrame({"date": dates, "dcoilwtico": 50.0})
    hol = pd.DataFrame({"date": [dates[150]], "type": ["Holiday"], "locale": ["National"],
                        "locale_name": ["Ecuador"], "description": ["x"], "transferred": [False]})
    return Snapshot(None, sales, promo, tx, oil, hol, stores)


def test_features_do_not_use_information_after_cutoff():
    snap = _snap()
    cutoff = pd.Timestamp("2017-05-01")
    f1 = cutoff_features(build_panel(snap, end=cutoff + pd.Timedelta(days=13)), cutoff, 14)
    leaked = snap.sales.copy()
    leaked.loc[pd.to_datetime(leaked["date"]) >= cutoff, "sales"] = 1e6
    snap2 = Snapshot(None, leaked, snap.promotions, snap.transactions, snap.oil, snap.holidays, snap.stores)
    f2 = cutoff_features(build_panel(snap2, end=cutoff + pd.Timedelta(days=13)), cutoff, 14)
    pd.testing.assert_frame_equal(f1[FEATURES], f2[FEATURES])


def test_dow_lags_point_to_same_weekday():
    snap = _snap()
    cutoff = pd.Timestamp("2017-05-01")
    f = cutoff_features(build_panel(snap, end=cutoff + pd.Timedelta(days=13)), cutoff, 14)
    row = f[(f["store_nbr"] == 1) & (f["family"] == "A") & (f["h"] == 3)].iloc[0]
    target = pd.Timestamp(row["date"])
    expected = ((target - pd.Timedelta(days=7) - pd.Timestamp("2017-01-01")).days % 7 + 1) * 10
    assert row["dow_lag_1"] == expected
    assert row["nat_holiday"] in (0, 1)


def test_models_return_ordered_non_negative_quantiles():
    snap = _snap()
    cutoff = pd.Timestamp("2017-05-01")
    panel = build_panel(snap, end=cutoff + pd.Timedelta(days=13))
    train = pd.concat([cutoff_features(panel, cutoff - pd.Timedelta(days=7 * k), 14, with_target=True)
                       for k in range(2, 8)])
    train = train[train["date"] < cutoff]
    test = cutoff_features(panel, cutoff, 14)
    q = [0.1, 0.5, 0.9]
    for m in (BaselineModel(q), LGBMQuantileModel(q, n_estimators=10, num_leaves=4, min_child_samples=2,
                                                   num_threads=1).fit(train)):
        p = m.predict(test).to_numpy()
        assert (p >= 0).all() and (np.diff(p, axis=1) >= -1e-9).all()


def test_missing_last_day_is_imputed_not_treated_as_zero():
    """Atraso/entrega parcial del domingo: el lunes no debe colapsar el pronóstico."""
    snap = _snap()
    cutoff = pd.Timestamp("2017-05-01")
    partial = snap.sales[pd.to_datetime(snap.sales["date"]) != cutoff - pd.Timedelta(days=1)]
    snap2 = Snapshot(None, partial, snap.promotions, snap.transactions, snap.oil, snap.holidays, snap.stores)
    full = cutoff_features(build_panel(snap, end=cutoff + pd.Timedelta(days=13)), cutoff, 14)
    miss = cutoff_features(build_panel(snap2, end=cutoff + pd.Timedelta(days=13)), cutoff, 14)
    assert miss["last_1_missing"].eq(1).all() and full["last_1_missing"].eq(0).all()
    assert miss["last_1"].notna().all()
    sunday = miss[miss["dow"] == 6]
    assert sunday["dow_lag_1_missing"].eq(1).all() and sunday["dow_lag_1_imp"].gt(0).all()
    # el nivel imputado es cercano al real (serie estable)
    assert (miss["dow_lag_1_imp"] / full["dow_lag_1_imp"]).between(0.5, 2).all()


def test_training_frame_simulates_missing_last_day_and_leading_zeros_are_unknown():
    from favorita.features import training_frame
    snap = _snap()
    cutoff = pd.Timestamp("2017-07-03")
    panel = build_panel(snap, end=cutoff + pd.Timedelta(days=13))
    tr = training_frame(panel, cutoff - pd.Timedelta(days=7), cutoff, 14, window_days=70, stride=7)
    assert tr["last_1_missing"].sum() > 0          # el modelo ve faltantes al entrenar
    assert tr["y"].notna().all()                   # el objetivo nunca se enmascara
    late = snap.sales.copy()
    late.loc[(late["store_nbr"] == 2) & (pd.to_datetime(late["date"]) < "2017-04-01"), "sales"] = 0.0
    snap3 = Snapshot(None, late, snap.promotions, snap.transactions, snap.oil, snap.holidays, snap.stores)
    f = cutoff_features(build_panel(snap3, end=cutoff + pd.Timedelta(days=13)), cutoff, 14)
    s2 = f[f["store_nbr"] == 2]
    assert (s2["mean_112"] > 0).all()               # los ceros previos a la apertura no cuentan
