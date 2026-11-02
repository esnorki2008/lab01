import pandas as pd

from favorita.io_utils import sha256_file
from favorita.lake import build_snapshot, ingest_until, list_inbox, load_asof
from favorita.source_system import build_source_system


def _inbox_checksums(s):
    return {e.path.as_posix(): sha256_file(e.path) for e in list_inbox(s)}


def test_source_system_is_deterministic_and_idempotent(project):
    m1 = build_source_system(project)
    c1 = _inbox_checksums(project)
    m2 = build_source_system(project)              # no regenera
    assert m1 == m2
    m3 = build_source_system(project, force=True)  # regenera igual
    assert _inbox_checksums(project) == c1 and m3["history_end"] == m1["history_end"]
    assert pd.Timestamp(m1["history_end"]).weekday() == 0


def test_history_only_contains_data_delivered_before_start(project):
    m = build_source_system(project)
    start = pd.Timestamp(m["history_end"])
    hist = pd.read_parquet(project.path("history_dir") / "sales.parquet")
    assert pd.to_datetime(hist["date"]).max() + pd.Timedelta(days=1) < start
    promo = pd.read_parquet(project.path("history_dir") / "promotions.parquet")
    # promociones llegan con 14 días de anticipación: la historia incluye el plan futuro conocido
    assert pd.to_datetime(promo["date"]).max() >= start


def test_inbox_has_one_file_per_date_and_source_with_anomalies(project):
    build_source_system(project)
    entries = list_inbox(project)
    assert {e.source for e in entries} >= {"sales", "transactions", "oil", "promotions"}
    names = [e.path.name for e in entries if e.source == "sales"]
    assert any(n.endswith("_v2.csv") for n in names)  # correcciones / partes / reenvíos


def test_ingest_is_idempotent_and_never_modifies_inbox(project):
    m = build_source_system(project)
    before = _inbox_checksums(project)
    day = pd.Timestamp(m["first_delivery"]) + pd.Timedelta(days=20)
    s1 = ingest_until(project, day)
    s2 = ingest_until(project, day)
    assert s1["files_processed_now"] > 0 and s2["files_processed_now"] == 0
    assert _inbox_checksums(project) == before


def test_asof_view_ignores_later_deliveries_and_applies_corrections(project):
    m = build_source_system(project)
    ingest_until(project, m["last_delivery"])
    corr = [e for e in list_inbox(project) if e.source == "sales" and e.path.name.endswith("_v2.csv")]
    assert corr, "la simulación debe incluir correcciones"
    e = corr[0]
    v2 = pd.read_csv(e.path)
    key = v2.iloc[0]
    before = load_asof(project, "sales", e.delivered - pd.Timedelta(days=1))
    after = load_asof(project, "sales", e.delivered)

    def val(df):
        r = df[(pd.to_datetime(df["date"]) == pd.Timestamp(key["date"])) & (df["store_nbr"] == key["store_nbr"])
               & (df["family"] == key["family"])]
        return r["sales"].tolist(), r["_version"].tolist()

    assert val(after) == ([float(key["sales"])], [2])
    assert val(before)[1] != [2]


def test_snapshot_never_contains_facts_on_or_after_cutoff(project):
    m = build_source_system(project)
    ingest_until(project, m["last_delivery"])
    cutoff = pd.Timestamp(m["first_delivery"]) + pd.Timedelta(days=14)
    snap = build_snapshot(project, cutoff, include_later_deliveries=True)
    assert pd.to_datetime(snap.sales["date"]).max() < cutoff
    assert pd.to_datetime(snap.transactions["date"]).max() < cutoff


def test_invalid_files_go_to_quarantine(project):
    m = build_source_system(project)
    ingest_until(project, m["last_delivery"])
    anomalies = pd.read_csv(project.path("reports_dir") / "source_system_anomalies.csv")
    invalid = anomalies[anomalies["note"].str.startswith("invalid:") & (anomalies["version"] == 1)]
    q = project.path("quarantine_dir")
    for _, r in invalid.iterrows():
        assert (q / r["file"]).exists() or (q / r["file"]).with_suffix(".csv.errors.json").exists() \
            or (q / (r["file"] + ".errors.json")).exists()
        lake_file = project.path("lake_dir") / r["file"].replace(".csv", ".parquet")
        assert not lake_file.exists()
