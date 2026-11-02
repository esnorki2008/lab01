"""Construcción de features por (serie, fecha de corte, horizonte).

La MISMA función genera las filas de entrenamiento (muchos cortes pasados) y
las de inferencia (corte actual): no hay divergencia entre entrenamiento y
producción. Para un corte ``c`` sólo se usan:

* ventas con fecha < c (lo conocido en c, ver ``lake.build_snapshot``),
* transacciones con fecha < c - lag_transacciones,
* precio del petróleo con fecha < c,
* plan de promociones y feriados de la fecha objetivo (se conocen con anticipación).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lake import Snapshot

DOW_LAGS = 8
CATEGORICAL = ["store_nbr", "family_code", "store_type_code", "cluster", "city_code"]


@dataclass
class Panel:
    """Matrices anchas (serie x día) a partir de un snapshot."""
    series: pd.DataFrame          # store_nbr, family + atributos de tienda
    dates: pd.DatetimeIndex
    sales: np.ndarray             # NaN = desconocido (no es lo mismo que 0)
    promo: np.ndarray
    store_tx: np.ndarray          # transacciones por serie (de su tienda)
    oil: np.ndarray               # por día, con forward-fill sólo hacia adelante
    nat_hol: np.ndarray           # por día
    reg_hol: np.ndarray           # serie x día
    loc_hol: np.ndarray           # serie x día
    workday: np.ndarray           # por día
    family_codes: dict[str, int]

    def pos(self, d: pd.Timestamp) -> int:
        return int((d - self.dates[0]).days)


def _code(values: pd.Series, mapping: dict | None = None) -> tuple[np.ndarray, dict]:
    if mapping is None:
        mapping = {v: i for i, v in enumerate(sorted(values.astype(str).unique()))}
    return values.astype(str).map(mapping).fillna(-1).astype(int).to_numpy(), mapping


def build_panel(snap: Snapshot, end: pd.Timestamp, family_codes: dict | None = None,
                start: pd.Timestamp | None = None) -> Panel:
    """Panel desde ``start`` hasta ``end`` inclusive (end puede estar en el futuro)."""
    sales = snap.sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    first = sales["date"].min() if start is None else start
    if pd.isna(first):
        first = end - pd.Timedelta(days=400)
    dates = pd.date_range(first, end, freq="D")
    series = sales[["store_nbr", "family"]].drop_duplicates()
    promo_df = snap.promotions.copy()
    promo_df["date"] = pd.to_datetime(promo_df["date"])
    series = pd.concat([series, promo_df[["store_nbr", "family"]]]).drop_duplicates()
    series = series.sort_values(["store_nbr", "family"]).reset_index(drop=True)
    stores = snap.stores.copy()
    stores["store_nbr"] = stores["store_nbr"].astype(int)
    series = series.merge(stores[["store_nbr", "city", "state", "type", "cluster"]], on="store_nbr", how="left")
    series["family_code"], fam_map = _code(series["family"], family_codes)
    series["store_type_code"], _ = _code(series["type"], {t: i for i, t in enumerate("ABCDE")})
    cities = sorted(stores["city"].astype(str).unique())
    series["city_code"], _ = _code(series["city"], {c: i for i, c in enumerate(cities)})
    series["cluster"] = pd.to_numeric(series["cluster"], errors="coerce").fillna(-1).astype(int)
    sid = pd.MultiIndex.from_frame(series[["store_nbr", "family"]])

    def wide(df: pd.DataFrame, col: str) -> np.ndarray:
        df = df[(df["date"] >= dates[0]) & (df["date"] <= dates[-1])]
        piv = df.pivot_table(index=["store_nbr", "family"], columns="date", values=col, aggfunc="last")
        piv = piv.reindex(index=sid, columns=dates)
        return piv.to_numpy(dtype=float)

    Y = wide(sales, "sales")
    # Antes de la primera venta (tienda que aún no abre / producto aún no vendido) el 0 no es
    # demanda observada: se trata como desconocido para no arrastrar los promedios hacia abajo.
    Y = mask_leading_zeros(Y)
    P = wide(promo_df, "onpromotion")

    tx = snap.transactions.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    tx = tx[(tx["date"] >= dates[0]) & (tx["date"] <= dates[-1])]
    txw = tx.pivot_table(index="store_nbr", columns="date", values="transactions", aggfunc="last")
    txw = txw.reindex(index=sorted(series["store_nbr"].unique()), columns=dates)
    store_pos = {s: i for i, s in enumerate(txw.index)}
    TX = txw.to_numpy(dtype=float)[[store_pos[s] for s in series["store_nbr"]]]

    oil = snap.oil.copy()
    oil["date"] = pd.to_datetime(oil["date"])
    oil_s = oil.set_index("date")["dcoilwtico"].astype(float).sort_index()
    oil_s = oil_s[~oil_s.index.duplicated(keep="last")]
    oil_full = oil_s.reindex(oil_s.index.union(dates)).ffill().reindex(dates)

    hol = snap.holidays.copy()
    hol["date"] = pd.to_datetime(hol["date"])
    hol["transferred"] = hol["transferred"].astype(str).str.lower().isin(["true", "1"])
    active = hol[~hol["transferred"] & hol["type"].isin(["Holiday", "Transfer", "Additional", "Bridge", "Event"])]
    dpos = {d: i for i, d in enumerate(dates)}
    nat = np.zeros(len(dates))
    work = np.zeros(len(dates))
    reg = np.zeros((len(series), len(dates)))
    loc = np.zeros((len(series), len(dates)))
    for _, h in active.iterrows():
        j = dpos.get(h["date"])
        if j is None:
            continue
        if h["locale"] == "National":
            nat[j] = 1
        elif h["locale"] == "Regional":
            reg[(series["state"] == h["locale_name"]).to_numpy(), j] = 1
        elif h["locale"] == "Local":
            loc[(series["city"] == h["locale_name"]).to_numpy(), j] = 1
    for d in hol.loc[hol["type"] == "Work Day", "date"]:
        j = dpos.get(d)
        if j is not None:
            work[j] = 1
    return Panel(series=series, dates=dates, sales=Y, promo=P, store_tx=TX, oil=oil_full.to_numpy(dtype=float),
                 nat_hol=nat, reg_hol=reg, loc_hol=loc, workday=work, family_codes=fam_map)


def mask_leading_zeros(Y: np.ndarray) -> np.ndarray:
    positive = np.nan_to_num(Y, nan=0.0) > 0
    started = np.cumsum(positive, axis=1) > 0
    return np.where(started, Y, np.nan)


def _nanmean(a: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        cnt = np.sum(~np.isnan(a), axis=1)
        s = np.nansum(a, axis=1)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def _nanstd(a: np.ndarray) -> np.ndarray:
    m = _nanmean(a)
    with np.errstate(all="ignore"):
        cnt = np.sum(~np.isnan(a), axis=1)
        v = np.nansum((a - m[:, None]) ** 2, axis=1) / np.maximum(cnt, 1)
    return np.where(cnt > 0, np.sqrt(v), np.nan)


def _window(M: np.ndarray, end: int, n: int) -> np.ndarray:
    lo = max(0, end - n)
    w = M[:, lo:end]
    if w.shape[1] < n:
        w = np.concatenate([np.full((M.shape[0], n - w.shape[1]), np.nan), w], axis=1)
    return w


def simulate_missing_last_day(panel: Panel, c: int, rng: np.random.Generator) -> np.ndarray:
    """Copia de las ventas con el día c-1 borrado como ocurre en producción.

    En producción, el lunes puede faltar el domingo completo (atraso) o una parte de las
    tiendas (entrega parcial). Si el entrenamiento sólo ve historia completa, el modelo
    aprende "dato faltante => serie muerta" y colapsa a ~0 justo en esas semanas.
    """
    Y = panel.sales.copy()
    u = rng.random()
    if u < 0.10:                       # atraso: falta el día completo
        Y[:, c - 1] = np.nan
    elif u < 0.40:                     # entrega parcial: falta ~30 % de las tiendas
        stores = panel.series["store_nbr"].to_numpy()
        uniq = np.unique(stores)
        miss = rng.choice(uniq, size=max(1, int(0.3 * len(uniq))), replace=False)
        Y[np.isin(stores, miss), c - 1] = np.nan
    return Y


def _cap_outliers(Y: np.ndarray, c: int, k: float = 4.0) -> np.ndarray:
    """Acota valores extremos de la historia (p. ej. un ×10 que pasó la validación) a k veces el
    percentil 95 de los últimos 112 días de la serie. Los picos reales (feriados) rara vez superan
    ese tope; uno erróneo no debe disparar el pedido de las semanas siguientes."""
    w = Y[:, max(0, c - 112):c]
    if w.size == 0:
        return Y
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            p95 = np.nanpercentile(w, 95, axis=1)
    cap = np.where(np.isnan(p95), np.inf, k * p95 + 10.0)
    return np.minimum(Y, cap[:, None]) if np.isfinite(cap).any() else Y


def cutoff_features(panel: Panel, cutoff: pd.Timestamp, horizon: int, tx_lag: int = 2,
                    with_target: bool = False, history: np.ndarray | None = None) -> pd.DataFrame:
    """Filas (serie x h) para un corte. Usa sólo columnas < cutoff para historia.

    ``history`` permite pasar una versión de las ventas con faltantes simulados (entrenamiento);
    el objetivo ``y`` siempre se toma del panel real.
    """
    c = panel.pos(cutoff)
    Y = panel.sales if history is None else history
    Y = _cap_outliers(Y, c)
    n = Y.shape[0]
    last = {}
    for k in (7, 14, 28, 56, 112):
        last[k] = _window(Y, c, k)
    mean_7 = _nanmean(last[7])
    last_1 = _window(Y, c, 1)[:, 0]
    feats_c = {
        "last_1_missing": np.isnan(last_1).astype(float),
        # imputación: dato faltante del último día => nivel de la última semana conocida
        "last_1": np.where(np.isnan(last_1), mean_7, last_1),
        "mean_7": mean_7, "mean_14": _nanmean(last[14]), "mean_28": _nanmean(last[28]),
        "mean_56": _nanmean(last[56]), "mean_112": _nanmean(last[112]),
        "std_28": _nanstd(last[28]),
        "zero_frac_28": _nanmean(np.where(np.isnan(last[28]), np.nan, (last[28] == 0).astype(float))),
        "known_days_7": np.sum(~np.isnan(last[7]), axis=1).astype(float),
        "promo_mean_28": _nanmean(_window(panel.promo, c, 28)),
        "tx_mean_28": _nanmean(_window(panel.store_tx, max(c - tx_lag + 1, 0), 28)),
        "oil_last": np.repeat(panel.oil[c - 1] if c >= 1 else np.nan, n),
        "oil_chg_28": np.repeat((panel.oil[c - 1] / panel.oil[c - 29] - 1) if c >= 29 else np.nan, n),
    }
    with np.errstate(all="ignore"):
        feats_c["trend_7_28"] = feats_c["mean_7"] / (feats_c["mean_28"] + 1e-3)
    # ventas de los últimos 8 días de la misma semana, por día de semana
    dow_hist = {}
    for dow in range(7):
        # último día < cutoff con este día de semana
        back = (cutoff.weekday() - dow) % 7 or 7
        cols = [c - back - 7 * i for i in range(DOW_LAGS)]
        mat = np.stack([Y[:, j] if j >= 0 else np.full(n, np.nan) for j in cols], axis=1)
        dow_hist[dow] = mat

    frames = []
    base = panel.series[["store_nbr", "family", "family_code", "store_type_code", "cluster", "city_code"]]
    for h in range(horizon):
        t = cutoff + pd.Timedelta(days=h)
        j = c + h
        dm = dow_hist[t.weekday()]
        f = base.copy()
        f["cutoff"] = cutoff
        f["date"] = t
        f["h"] = h
        f["dow"] = t.weekday()
        f["dom"] = t.day
        f["month"] = t.month
        f["payday"] = int(t.day == 15 or (t + pd.Timedelta(days=1)).day == 1)
        for k, v in feats_c.items():
            f[k] = v
        dow_mean_8 = _nanmean(dm)
        for i in range(DOW_LAGS):
            f[f"dow_lag_{i + 1}"] = dm[:, i]
        # el modelo usa los 2 rezagos más recientes: si faltan, se imputan con la media del mismo día
        f["dow_lag_1_missing"] = np.isnan(dm[:, 0]).astype(float)
        for i in (0, 1):
            f[f"dow_lag_{i + 1}_imp"] = np.where(np.isnan(dm[:, i]), dow_mean_8, dm[:, i])
        f["dow_mean_4"] = _nanmean(dm[:, :4])
        f["dow_mean_8"] = dow_mean_8
        ly = j - 364
        f["ly_sales"] = Y[:, ly] if 0 <= ly < c else np.nan
        f["onpromotion"] = panel.promo[:, j] if j < panel.promo.shape[1] else np.nan
        f["nat_holiday"] = panel.nat_hol[j] if j < len(panel.nat_hol) else 0
        f["reg_holiday"] = panel.reg_hol[:, j] if j < panel.reg_hol.shape[1] else 0
        f["loc_holiday"] = panel.loc_hol[:, j] if j < panel.loc_hol.shape[1] else 0
        f["workday_event"] = panel.workday[j] if j < len(panel.workday) else 0
        if with_target:
            f["y"] = panel.sales[:, j] if j < panel.sales.shape[1] else np.nan
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


FEATURES = [
    "store_nbr", "family_code", "store_type_code", "cluster", "city_code",
    "h", "dow", "dom", "month", "payday",
    "last_1", "mean_7", "mean_14", "mean_28", "mean_56", "mean_112", "std_28", "zero_frac_28",
    "known_days_7", "last_1_missing", "trend_7_28", "promo_mean_28", "tx_mean_28", "oil_last", "oil_chg_28",
    "dow_lag_1_imp", "dow_lag_2_imp", "dow_lag_1_missing", "dow_mean_4", "dow_mean_8", "ly_sales",
    "onpromotion", "nat_holiday", "reg_holiday", "loc_holiday", "workday_event",
]


def training_frame(panel: Panel, last_cutoff: pd.Timestamp, known_until: pd.Timestamp, horizon: int,
                   window_days: int, stride: int, tx_lag: int = 2, simulate_missing: bool = True,
                   seed: int = 0) -> pd.DataFrame:
    """Filas de entrenamiento con cortes <= last_cutoff y objetivo < known_until.

    Con ``simulate_missing`` se reproducen en la historia los faltantes del último día que
    ocurren en producción (determinístico por semilla y fecha de corte).
    """
    cutoffs = []
    c = last_cutoff
    earliest = panel.dates[0] + pd.Timedelta(days=120)
    while c >= last_cutoff - pd.Timedelta(days=window_days) and c >= earliest:
        cutoffs.append(c)
        c -= pd.Timedelta(days=stride)
    frames = []
    for c in sorted(cutoffs):
        hist = None
        if simulate_missing:
            rng = np.random.default_rng([seed, int(c.strftime("%Y%m%d"))])
            hist = simulate_missing_last_day(panel, panel.pos(c), rng)
        frames.append(cutoff_features(panel, c, horizon, tx_lag, with_target=True, history=hist))
    if not frames:
        return pd.DataFrame(columns=FEATURES + ["y"])
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["date"] < known_until) & df["y"].notna()]
    return df.reset_index(drop=True)
