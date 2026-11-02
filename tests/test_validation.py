import pandas as pd

from favorita.validation import ValidationContext, validate_frame

D = pd.Timestamp("2017-03-01")


def sales(n=40, **over):
    df = pd.DataFrame({"date": ["2017-03-01"] * n, "store_nbr": [str(1 + i % 4) for i in range(n)],
                       "family": [f"F{i // 4}" for i in range(n)], "sales": ["10.5"] * n})
    for k, v in over.items():
        df[k] = v
    return df


def ctx(**kw):
    base = dict(data_date=D, known_stores={1, 2, 3, 4}, known_families={f"F{i}" for i in range(30)},
                max_bad_row_frac=0.05)
    base.update(kw)
    return ValidationContext(**base)


def test_valid_file_accepted_and_typed():
    r = validate_frame(sales(), "sales", ctx())
    assert r.ok and len(r.clean) == 40 and len(r.rejected) == 0
    assert r.clean["sales"].dtype == float and r.clean["store_nbr"].dtype == "int64"


def test_missing_column_quarantines_file():
    r = validate_frame(sales().drop(columns="sales"), "sales", ctx())
    assert not r.ok and "columnas faltantes" in r.errors[0]


def test_empty_file_quarantined_but_empty_holidays_ok():
    assert not validate_frame(sales().iloc[0:0], "sales", ctx()).ok
    hol = pd.DataFrame(columns=["date", "type", "locale", "locale_name", "description", "transferred"])
    assert validate_frame(hol, "holidays", ctx(granularity="month")).ok


def test_few_bad_rows_are_rejected_individually():
    df = sales()
    df.loc[0, "sales"] = "-3"
    r = validate_frame(df, "sales", ctx(max_bad_row_frac=0.05))
    assert r.ok and len(r.clean) == 39
    assert r.rejected["_reason"].tolist() == ["negative_sales"]


def test_many_bad_rows_quarantine_whole_file():
    df = sales()
    df.loc[:10, "sales"] = "#N/A"
    r = validate_frame(df, "sales", ctx())
    assert not r.ok and len(r.clean) == 0


def test_date_mismatch_and_unknown_master_data():
    df = sales(n=100)
    df.loc[0, "date"] = "2017-02-28"
    df.loc[1, "store_nbr"] = "99"
    df.loc[2, "family"] = "NUEVA"
    r = validate_frame(df, "sales", ctx())
    assert r.ok
    assert set(r.rejected["_reason"]) == {"date_mismatch", "unknown_store", "unknown_family"}


def test_identical_duplicates_dropped_conflicting_rejected():
    df = sales(n=100)
    dup_same = df.iloc[[0]]
    dup_conflict = df.iloc[[1]].assign(sales="99")
    r = validate_frame(pd.concat([df, dup_same, dup_conflict], ignore_index=True), "sales", ctx())
    assert r.ok
    assert len(r.clean) == 99  # 100 - 1 llave en conflicto
    assert (r.rejected["_reason"] == "conflicting_duplicate").sum() == 2
    assert any("duplicadas" in w for w in r.warnings)


def test_oil_allows_missing_price_but_not_text():
    oil = pd.DataFrame({"date": ["2017-03-01"], "dcoilwtico": [""]})
    r = validate_frame(oil, "oil", ctx())
    assert r.ok and r.clean["dcoilwtico"].isna().all()
    assert not validate_frame(oil.assign(dcoilwtico="abc"), "oil", ctx()).ok
