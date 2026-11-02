"""Calendario de la simulación para los DAGs (sólo lee JSON/YAML: barato al parsear)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from .config import load_settings


def _delivery_folders(inbox) -> list[datetime]:
    out = []
    if not inbox.exists():
        return out
    for src in inbox.iterdir():
        if src.is_dir():
            for d in src.iterdir():
                if d.is_dir() and d.name.startswith("delivered="):
                    try:
                        out.append(datetime.fromisoformat(d.name.split("=", 1)[1]))
                    except ValueError:
                        continue
    return out


def simulation_window() -> tuple[datetime, datetime] | None:
    """(primer día, último día) de operación: manifiesto de la bandeja y/o carpetas de entrega.

    Si se agregan a la bandeja entregas posteriores (p. ej. un periodo nuevo),
    el calendario de los DAGs se extiende automáticamente hasta la última.
    """
    try:
        settings = load_settings()
        inbox = settings.path("inbox_dir")
    except (OSError, ValueError, KeyError):
        return None
    starts, ends = [], []
    for mpath in (inbox / "_manifest.json", settings.path("history_dir") / "_manifest.json"):
        try:
            with open(mpath, encoding="utf-8") as fh:
                m = json.load(fh)
            starts.append(datetime.fromisoformat(m["first_delivery"]))
            ends.append(datetime.fromisoformat(m["last_delivery"]))
            break
        except (OSError, ValueError, KeyError):
            continue
    folders = _delivery_folders(inbox)
    if folders:
        if not starts:
            starts.append(min(folders))
        ends.append(max(folders))
    if not starts:
        return None
    return starts[0].replace(tzinfo=timezone.utc), max(ends).replace(tzinfo=timezone.utc)


def first_monday(d: datetime) -> datetime:
    return d + timedelta(days=(7 - d.weekday()) % 7)
