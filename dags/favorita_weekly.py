"""Pronóstico y pedido sugerido semanal (fecha de corte = lunes).

wait_for_ingest -> prepare_features -> train_or_reuse -> forecast -> suggest_order -> publish -> dvc_snapshot

* Sólo usa información entregada hasta la fecha de corte (vista as-of del lago).
* ``train_or_reuse`` reentrena cada ``retrain_every_weeks`` y aplica el criterio
  champion/challenger; las demás semanas reutiliza el modelo vigente.
* Param ``restate=true`` re-expresa el corte con correcciones llegadas después;
  el resultado se guarda aparte y el pronóstico oficial no se modifica.
* Datos insuficientes/desactualizados (incidente) => la tarea falla y no se publica.
"""
from __future__ import annotations

from datetime import timedelta

from airflow.sdk import Param, dag, get_current_context, task

from favorita_common import DEFAULT_ARGS, run_date, settings, window

SCHEDULE, START, END = window(weekly=True)


def _restate() -> bool:
    return bool(get_current_context()["params"].get("restate"))


@dag(
    dag_id="favorita_weekly_forecast",
    schedule=SCHEDULE,
    start_date=START,
    end_date=END,
    catchup=True,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args=DEFAULT_ARGS,
    params={"run_date": Param(None, type=["null", "string"], description="YYYY-MM-DD (lunes)"),
            "restate": Param(False, type="boolean")},
    tags=["favorita", "pronostico"],
    doc_md=__doc__,
)
def favorita_weekly_forecast():
    @task.sensor(poke_interval=30, timeout=60 * 60 * 24, mode="reschedule")
    def wait_for_ingest() -> bool:
        from favorita.lake import ingest_done
        return ingest_done(settings(), run_date(get_current_context(), weekly=True))

    @task(retries=0)
    def prepare_features() -> dict:
        from favorita.forecasting import prepare_features as prep
        return prep(settings(), run_date(get_current_context(), weekly=True), restate=_restate())

    @task(execution_timeout=timedelta(hours=2))
    def train_or_reuse() -> dict:
        if _restate():
            return {"trained": False, "reason": "re-expresión: se usa el modelo vigente en el corte"}
        from favorita.pipeline import first_cutoff
        from favorita.training import train_and_register
        s = settings()
        r = train_and_register(s, run_date(get_current_context(), weekly=True), first_cutoff=first_cutoff(s))
        e = r.get("entry") or {}
        return {"trained": r["trained"], "champion": e.get("champion"), "reason": e.get("reason")}

    @task
    def forecast() -> dict:
        from favorita.forecasting import generate_forecast
        return generate_forecast(settings(), run_date(get_current_context(), weekly=True), restate=_restate())

    @task
    def suggest_order() -> dict:
        from favorita.forecasting import build_orders
        return build_orders(settings(), run_date(get_current_context(), weekly=True), restate=_restate())

    @task
    def publish() -> dict:
        from favorita.forecasting import publish as pub
        m = pub(settings(), run_date(get_current_context(), weekly=True), restate=_restate())
        return {k: m.get(k) for k in ("cutoff", "kind", "status", "hashes")}

    @task(retries=0)
    def dvc_snapshot() -> dict:
        from favorita.versioning import dvc_snapshot as snap
        return snap(settings())

    (wait_for_ingest() >> prepare_features() >> train_or_reuse() >> forecast()
     >> suggest_order() >> publish() >> dvc_snapshot())


favorita_weekly_forecast()
