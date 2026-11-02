"""Utilidades compartidas por los DAGs (sólo orquestación, nada de lógica de negocio)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from favorita.schedule import first_monday, simulation_window

DEFAULT_ARGS = {
    "owner": "favorita",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

PLACEHOLDER_START = datetime(2017, 1, 2, tzinfo=timezone.utc)


def window(weekly: bool = False) -> tuple[str | None, datetime, datetime | None]:
    """(schedule, start, end) a partir del manifiesto de la bandeja.

    Mientras el sistema de origen no se haya construido el DAG queda sin
    calendario; cuando aparece el manifiesto, el scheduler re-parsea el archivo
    y el DAG empieza a correr (catchup) desde el primer día de la bandeja.
    """
    w = simulation_window()
    if w is None:
        return None, PLACEHOLDER_START, None
    start, end = w
    if weekly:
        return "0 0 * * 1", first_monday(start), end
    return "@daily", start, end


def run_date(context, weekly: bool = False) -> str:
    """Fecha de negocio de la ejecución: param explícito > logical_date (corridas programadas).

    Un disparo manual sin ``run_date`` NO usa la fecha de hoy (quedaría esperando datos que nunca
    llegan): falla de inmediato con un mensaje claro.
    """
    from favorita.validation import parse_business_date
    params = context.get("params") or {}
    dag_run = context.get("dag_run")
    if params.get("run_date"):
        value = str(params["run_date"])
    elif context.get("logical_date") is not None and getattr(dag_run, "run_type", "scheduled") != "manual":
        value = context["logical_date"].strftime("%Y-%m-%d")
    else:
        raise ValueError("Disparo manual sin fecha: use --conf '{\"run_date\": \"YYYY-MM-DD\"}'")
    return parse_business_date(value, monday=weekly)


def settings():
    from favorita.config import load_settings
    return load_settings()
