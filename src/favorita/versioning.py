"""Versionado de datos con DVC desde el flujo (``dvc add`` + ``dvc push``)."""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys

from .config import Settings

log = logging.getLogger(__name__)


def _dvc_cmd() -> list[str] | None:
    if shutil.which("dvc"):
        return ["dvc"]
    try:
        import dvc  # noqa: F401
        return [sys.executable, "-m", "dvc"]
    except ImportError:
        return None


def dvc_snapshot(settings: Settings, push: bool = True) -> dict:
    """Registra en la caché de DVC el estado actual de datos, modelos y pronósticos.

    Los archivos ``*.dvc`` resultantes se commitean a Git (ver README) para fijar
    la versión. Un fallo aquí no invalida el pronóstico ya publicado: se registra
    como error en el log y en el resultado de la tarea.
    """
    if not settings["versioning"].get("dvc_auto_snapshot", True):
        return {"status": "disabled"}
    cmd = _dvc_cmd()
    root = settings.root
    if cmd is None or not (root / ".dvc").exists():
        log.warning("DVC no disponible o repositorio sin `dvc init`; se omite el snapshot.")
        return {"status": "skipped"}
    targets = [t for t in settings["versioning"]["dvc_targets"] if (root / t).exists()]
    if not targets:
        return {"status": "nothing"}
    res = subprocess.run([*cmd, "add", "--quiet", *targets], cwd=root, capture_output=True, text=True)
    if res.returncode != 0:
        # No se revierte una publicación válida por un problema de versionado; se reporta.
        log.error("dvc add falló: %s", res.stderr.strip())
        return {"status": "error", "stderr": res.stderr.strip()[-2000:]}
    pushed = False
    if push:
        r = subprocess.run([*cmd, "push", "--quiet", *[f"{t}.dvc" for t in targets]], cwd=root,
                           capture_output=True, text=True)
        if r.returncode != 0:
            log.warning("dvc push falló (¿remoto configurado?): %s", r.stderr.strip())
        pushed = r.returncode == 0
    return {"status": "ok", "targets": targets, "pushed": pushed}
