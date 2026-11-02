"""Reprocesamiento manual de un periodo con datos corregidos.

1. Ingiere todo lo entregado hasta ``end_date`` (correcciones incluidas).
2. Re-expresa cada corte del periodo con la información corregida
   (se guarda en ``data/forecasts/restated``; lo publicado no se toca).
3. Recalcula el seguimiento de desempeño al ``end_date``.

Para re-ejecutar el flujo normal sobre fechas pasadas use backfill
(ver README): las tareas son idempotentes y reproducen lo ya publicado.
"""
from __future__ import annotations

from datetime import datetime, timezone

from airflow.sdk import Param, dag, get_current_context, task

from favorita_common import DEFAULT_ARGS, settings


@dag(
    dag_id="favorita_reprocess",
    schedule=None,
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    is_paused_upon_creation=False,
    default_args=DEFAULT_ARGS,
    params={"start_date": Param(type="string", format="date", description="Primer corte (YYYY-MM-DD)"),
            "end_date": Param(type="string", format="date", description="Último día (YYYY-MM-DD)")},
    tags=["favorita", "reproceso"],
    doc_md=__doc__,
)
def favorita_reprocess():
    @task
    def ingest_corrections() -> dict:
        from favorita.lake import ingest_until
        from favorita.validation import parse_business_date
        s = ingest_until(settings(), parse_business_date(get_current_context()["params"]["end_date"]))
        return {"processed": s["files_processed_now"]}

    @task
    def list_cutoffs() -> list[str]:
        import pandas as pd
        p = get_current_context()["params"]
        s = settings()
        wd = int(s["forecast"]["cutoff_weekday"])
        days = pd.date_range(p["start_date"], p["end_date"], freq="D")
        return [f"{d:%Y-%m-%d}" for d in days if d.weekday() == wd]

    @task(max_active_tis_per_dag=2)
    def restate_cutoff(cutoff: str) -> dict:
        from favorita.pipeline import weekly
        r = weekly(settings(), cutoff, restate=True)
        return {"cutoff": cutoff, "path": r["publish"].get("path")}

    @task(trigger_rule="all_done")
    def refresh_monitoring() -> dict:
        from favorita.monitoring import run_monitoring
        return run_monitoring(settings(), get_current_context()["params"]["end_date"])

    cuts = list_cutoffs()
    ingest_corrections() >> cuts
    restate_cutoff.expand(cutoff=cuts) >> refresh_monitoring()


favorita_reprocess()
