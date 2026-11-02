"""Los DAGs sólo orquestan: sin lógica de datos a nivel de módulo y con import válido."""
import ast
from pathlib import Path

import pytest

DAGS = Path(__file__).resolve().parents[1] / "dags"
DAG_FILES = sorted(p for p in DAGS.glob("favorita_*.py") if p.name != "favorita_common.py")
FORBIDDEN_TOP_LEVEL = {"pandas", "numpy", "lightgbm", "sklearn", "pyarrow"}


@pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.name)
def test_dag_files_do_not_import_data_libraries_at_top_level(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(m.split(".")[0] in FORBIDDEN_TOP_LEVEL for m in mods), f"{path.name}: {mods}"


@pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.name)
def test_dag_tasks_delegate_to_favorita_modules(path):
    src = path.read_text(encoding="utf-8")
    assert "from favorita" in src
    assert "read_parquet" not in src and "read_csv" not in src


def test_expected_dags_exist():
    names = {p.stem for p in DAG_FILES}
    assert {"favorita_source_system", "favorita_ingest", "favorita_weekly", "favorita_monitoring",
            "favorita_reprocess"} <= names


def test_dagbag_imports_without_errors():
    pytest.importorskip("airflow")
    from airflow.models import DagBag
    bag = DagBag(dag_folder=str(DAGS), include_examples=False)
    assert not bag.import_errors, bag.import_errors
    assert {"favorita_ingest", "favorita_weekly_forecast", "favorita_monitoring", "favorita_reprocess",
            "favorita_source_system"} <= set(bag.dag_ids)
