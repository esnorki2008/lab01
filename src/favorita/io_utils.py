"""Utilidades de E/S: escrituras atómicas (idempotencia) y hashes."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    """Hash estable del contenido de un DataFrame (independiente del formato de archivo)."""
    canon = df.reset_index(drop=True)
    canon = canon[sorted(canon.columns)]
    # lineterminator fijo: el hash debe ser idéntico en Windows y Linux
    data = canon.to_csv(index=False, float_format="%.6f", lineterminator="\n").encode()
    return hashlib.sha256(data).hexdigest()


def _atomic_replace(tmp: str, dest: Path) -> None:
    os.replace(tmp, dest)


def normalize_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica la resolución de fechas a ns (pandas 2 puede crear s/ms según el origen)."""
    cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not cols:
        return df
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype("datetime64[ns]")
    return df


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    df = normalize_datetimes(df)
    try:
        df.to_parquet(tmp, index=False)
        _atomic_replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def write_csv(df: pd.DataFrame, path: str | Path, **kw: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False, lineterminator="\n", **kw)
        _atomic_replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def write_json(obj: Any, path: str | Path) -> Path:
    return write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), path)


def write_text(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    try:
        _atomic_replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(settings) -> dict | None:
    """Manifiesto del periodo operado: en la bandeja (simulada) o junto a la historia (bandeja externa)."""
    for p in (settings.path("inbox_dir") / "_manifest.json", settings.path("history_dir") / "_manifest.json"):
        m = read_json(p)
        if m:
            return m
    return None


def code_fingerprint(root: str | Path) -> str:
    """Hash del código y la configuración que producen resultados (independiente de Git y del SO)."""
    root = Path(root)
    files = sorted([*root.glob("src/favorita/**/*.py"), *root.glob("dags/*.py"), root / "config" / "params.yaml"])
    h = hashlib.sha256()
    for f in files:
        if f.exists():
            h.update(f.relative_to(root).as_posix().encode())
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def code_is_dirty(root: str | Path) -> bool | None:
    """True si hay cambios sin commitear en código/config (None si no se puede determinar)."""
    import shutil
    import subprocess
    if not shutil.which("git"):
        return None
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--", "src", "dags", "config"], cwd=root,
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(r.stdout.strip()) if r.returncode == 0 else None


def code_version(root: str | Path) -> str:
    """SHA del commit de Git sin depender del binario `git` (útil dentro de Docker)."""
    env = os.environ.get("FAVORITA_CODE_VERSION")
    if env:
        return env
    git = Path(root) / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1]
        ref_file = git / ref
        if ref_file.exists():
            return ref_file.read_text().strip()
        packed = git / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(ref):
                    return line.split(" ")[0]
    except OSError:
        pass
    return "unknown"
