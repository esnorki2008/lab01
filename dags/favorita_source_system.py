"""Sistema de origen simulado (representa a la empresa, NO al producto).

Construye una sola vez la carga histórica y la bandeja a partir de los
archivos de Kaggle. Es idempotente: si los archivos de Kaggle y la configuración
no cambiaron, no hace nada.
"""
from __future__ import annotations

from datetime import datetime, timezone

from airflow.sdk import dag, task

from favorita_common import DEFAULT_ARGS, settings


@dag(
    dag_id="favorita_source_system",
    schedule="@once",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    is_paused_upon_creation=False,
    default_args=DEFAULT_ARGS,
    tags=["favorita", "simulacion"],
    doc_md=__doc__,
)
def favorita_source_system():
    @task
    def build_source_system() -> dict:
        from favorita.source_system import build_source_system as build
        m = build(settings())
        return {k: m.get(k) for k in ("history_end", "first_delivery", "last_delivery", "n_files", "external_inbox")}

    build_source_system()


favorita_source_system()
