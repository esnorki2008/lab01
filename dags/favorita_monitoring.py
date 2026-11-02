"""Seguimiento diario del desempeño de los pronósticos publicados.

Compara cada pronóstico oficial con las ventas reales conocidas a la fecha
(incluidas correcciones). Se recalcula completo: es idempotente y refleja las
correcciones automáticamente. Genera ``data/monitoring/*`` y ``reports/monitoring_report.md``.
"""
from __future__ import annotations

from airflow.sdk import dag, get_current_context, task

from favorita_common import DEFAULT_ARGS, run_date, settings, window

SCHEDULE, START, END = window()


@dag(
    dag_id="favorita_monitoring",
    schedule=SCHEDULE,
    start_date=START,
    end_date=END,
    catchup=True,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args=DEFAULT_ARGS,
    params={"run_date": None},
    tags=["favorita", "monitoreo"],
    doc_md=__doc__,
)
def favorita_monitoring():
    @task.sensor(poke_interval=30, timeout=60 * 60 * 24, mode="reschedule")
    def wait_for_ingest() -> bool:
        from favorita.lake import ingest_done
        return ingest_done(settings(), run_date(get_current_context()))

    @task
    def evaluate_published_forecasts() -> dict:
        from favorita.monitoring import run_monitoring
        return run_monitoring(settings(), run_date(get_current_context()))

    wait_for_ingest() >> evaluate_published_forecasts()


favorita_monitoring()
