"""Ingesta y validación diaria de las fuentes.

Para la fecha de ejecución D:
1. Carga histórica inicial (una vez; idempotente por checksum).
2. Lee de la bandeja (sin modificarla) todo lo entregado hasta D que aún no se
   haya procesado; valida cada archivo; lo aceptado va al lago y lo inválido
   a cuarentena. Archivos tardíos, parciales o corregidos se procesan cuando llegan.
3. Deja un marcador ``_runs/ingest_D.json`` que esperan los demás DAGs.
"""
from __future__ import annotations

from airflow.sdk import dag, get_current_context, task

from favorita_common import DEFAULT_ARGS, run_date, settings, window

SCHEDULE, START, END = window()


@dag(
    dag_id="favorita_ingest",
    schedule=SCHEDULE,
    start_date=START,
    end_date=END,
    catchup=True,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args=DEFAULT_ARGS,
    params={"run_date": None},
    tags=["favorita", "ingesta"],
    doc_md=__doc__,
)
def favorita_ingest():
    @task
    def load_history() -> dict:
        from favorita.lake import load_history as load
        s = load(settings())
        return {k: v.get("rows_accepted") for k, v in s.get("sources", {}).items()}

    @task
    def ingest_and_validate() -> dict:
        from favorita.lake import ingest_until
        summary = ingest_until(settings(), run_date(get_current_context()))
        return {k: summary[k] for k in ("run_date", "files_processed_now", "accepted_now",
                                        "quarantined_now", "warnings", "quarantined_files")}

    load_history() >> ingest_and_validate()


favorita_ingest()
