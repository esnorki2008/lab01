"""Escenarios de operación reportados por la evaluación (periodo no publicado)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from favorita.lake import ingest_until, list_inbox, load_asof
from favorita.source_system import build_source_system, operating_range
from favorita.validation import ValidationContext, validate_frame


def _first_sales(project):
    return next(e for e in list_inbox(project) if e.source == "sales" and e.path.name.endswith("_v1.csv"))


def test_late_old_version_does_not_override_newer_one(project):
    m = build_source_system(project)
    e = _first_sales(project)
    data_date = e.path.name.split("_")[1]
    good = pd.read_csv(e.path)
    inbox = project.path("inbox_dir") / "sales"
    d1 = pd.Timestamp(m["first_delivery"]) + pd.Timedelta(days=20)
    v2 = good.copy()
    v2["sales"] = 123.0
    (inbox / f"delivered={d1:%Y-%m-%d}").mkdir(parents=True, exist_ok=True)
    v2.to_csv(inbox / f"delivered={d1:%Y-%m-%d}" / f"sales_{data_date}_v7.csv", index=False)
    stale = good.assign(sales=999.0)
    d2 = d1 + pd.Timedelta(days=1)
    (inbox / f"delivered={d2:%Y-%m-%d}").mkdir(parents=True, exist_ok=True)
    stale.to_csv(inbox / f"delivered={d2:%Y-%m-%d}" / f"sales_{data_date}_v3.csv", index=False)
    ingest_until(project, d2)
    view = load_asof(project, "sales", d2)
    row = view[pd.to_datetime(view["date"]) == pd.Timestamp(data_date)]
    assert (row["sales"] == 123.0).all() and (row["_version"] == 7).all()


def test_unknown_master_rows_do_not_discard_valid_correction():
    df = pd.DataFrame({"date": ["2017-03-01"] * 10, "store_nbr": ["1"] * 8 + ["55", "56"],
                       "family": [f"F{i}" for i in range(10)], "sales": ["5"] * 10})
    ctx = ValidationContext(data_date=pd.Timestamp("2017-03-01"), known_stores={1},
                            known_families={f"F{i}" for i in range(10)})
    r = validate_frame(df, "sales", ctx)
    assert r.ok and len(r.clean) == 8 and set(r.rejected["_reason"]) == {"unknown_store"}


def test_small_file_with_one_bad_row_is_not_fully_rejected():
    df = pd.DataFrame({"date": ["2017-03-01"] * 6, "store_nbr": ["1"] * 6,
                       "family": [f"F{i}" for i in range(6)], "sales": ["5", "5", "5", "5", "5", "-1"]})
    r = validate_frame(df, "sales", ValidationContext(data_date=pd.Timestamp("2017-03-01")))
    assert r.ok and len(r.clean) == 5


def test_tolerant_formats_semicolon_uppercase_datetime(project, tmp_path):
    from favorita.lake import read_source_csv
    p = tmp_path / "x.csv"
    p.write_text("﻿Date;Store_Nbr;Family;Sales\r\n2017-03-01 00:00:00;1;beverages;3.5\r\n", encoding="utf-8")
    r = validate_frame(read_source_csv(p), "sales", ValidationContext(data_date=pd.Timestamp("2017-03-01"),
                                                                       known_families={"BEVERAGES"}))
    assert r.ok and r.clean["family"].tolist() == ["BEVERAGES"] and r.clean["sales"].tolist() == [3.5]


def test_isolated_typo_date_does_not_move_operation():
    dates = pd.to_datetime(["2017-01-01"] * 3 + ["2017-01-02"] * 3 + ["2071-08-10"])
    stores = pd.Series([1, 2, 3, 1, 2, 3, 1])
    lo, hi = operating_range(pd.Series(dates), stores)
    assert hi == pd.Timestamp("2017-01-02")


def test_short_train_keeps_minimum_history(tmp_path):
    from conftest import make_project
    s = make_project(tmp_path, overrides={"source_system": {"inbox_days": 400}}, days=300)
    m = build_source_system(s)
    hist = pd.read_parquet(s.path("history_dir") / "sales.parquet")
    assert len(hist) > 0
    assert (pd.Timestamp(m["history_end"]) - pd.to_datetime(hist["date"]).min()).days >= 150


def test_external_inbox_is_not_modified_and_history_is_built(tmp_path):
    from conftest import make_project
    from favorita.io_utils import sha256_file
    s = make_project(tmp_path, days=300)
    ext = s.path("inbox_dir") / "sales" / "delivered=2017-08-01"
    ext.mkdir(parents=True)
    pd.DataFrame({"date": ["2017-07-31"], "store_nbr": [1], "family": ["DAIRY"], "sales": [3.0]}).to_csv(
        ext / "sales_2017-07-31_v1.csv", index=False)
    before = sha256_file(ext / "sales_2017-07-31_v1.csv")
    m = build_source_system(s)
    assert m["external_inbox"] and m["first_delivery"] == "2017-08-01"
    assert sha256_file(ext / "sales_2017-07-31_v1.csv") == before
    assert not (s.path("inbox_dir") / "_manifest.json").exists()
    summary = ingest_until(s, "2017-08-01")
    assert summary["accepted_now"] == 1


def test_manual_trigger_without_run_date_fails_fast():
    from favorita_common import run_date
    manual = {"params": {}, "logical_date": pd.Timestamp("2026-11-02"),
              "dag_run": SimpleNamespace(run_type="manual")}
    with pytest.raises(ValueError):
        run_date(manual)
    assert run_date({"params": {"run_date": "2017-03-06"}, "dag_run": SimpleNamespace(run_type="manual")},
                    weekly=True) == "2017-03-06"
    with pytest.raises(ValueError):
        run_date({"params": {"run_date": "2017-03-07"}}, weekly=True)  # no es lunes
    sched = {"params": {}, "logical_date": pd.Timestamp("2017-03-07"),
             "dag_run": SimpleNamespace(run_type="scheduled")}
    assert run_date(sched) == "2017-03-07"


def test_chain_closure_days_inferred_from_history():
    from favorita.forecasting import chain_closure_days
    d = pd.date_range("2016-12-20", "2016-12-31").drop(pd.Timestamp("2016-12-25"))
    sales = pd.DataFrame({"date": d, "store_nbr": 1, "family": "A", "sales": 5.0})
    assert chain_closure_days(sales) == ["12-25"]
