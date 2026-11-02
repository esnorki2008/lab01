"""Ingesta (bandeja -> lago) y vistas *as-of* del lago.

Lago (append-only, nunca se sobreescribe con otra cosa que el mismo contenido):

    data/lake/<fuente>/history.parquet                       carga inicial
    data/lake/<fuente>/delivered=YYYY-MM-DD/<archivo>.parquet entregas aceptadas
    data/lake/_registry/<fuente>/delivered=.../<archivo>.json  qué se procesó (checksum, estado)
    data/lake/_runs/ingest_YYYY-MM-DD.json                     resumen de cada ejecución

La vista *as-of T* combina la carga inicial con todas las entregas con
``delivered <= T`` y, para cada llave, toma el valor de la entrega más
reciente (upsert por fila). Así una corrección sólo afecta a las vistas
posteriores a su llegada, y cualquier pronóstico pasado se puede reconstruir.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import Settings
from .io_utils import load_manifest, read_json, sha256_file, write_csv, write_json, write_parquet
from .schemas import SCHEMAS
from .validation import ValidationContext, validate_frame

log = logging.getLogger(__name__)

SOURCE_ORDER = ("stores", "holidays", "oil", "promotions", "transactions", "sales")
FILE_RE = re.compile(r"^(?P<source>[a-z]+)_(?P<date>\d{4}-\d{2}-\d{2})_v(?P<version>\d+)\.csv$", re.IGNORECASE)
DELIVERED_RE = re.compile(r"^delivered=(\d{4}-\d{2}-\d{2})$")
META_COLS = ["_delivered", "_version", "_file"]


def _granularity(settings: Settings, source: str) -> str:
    return settings["source_system"]["sources"].get(source, {}).get("granularity", "day")


def _ctx(settings: Settings, **kw) -> ValidationContext:
    v = settings["validation"]
    return ValidationContext(max_bad_row_frac=float(v["max_bad_row_frac"]),
                             max_sales_value=float(v["max_sales_value"]), **kw)


# ----------------------------------------------------------------------------
# Carga histórica inicial
# ----------------------------------------------------------------------------
def load_history(settings: Settings) -> dict:
    hist_dir = settings.path("history_dir")
    lake = settings.path("lake_dir")
    quarantine = settings.path("quarantine_dir")
    manifest = load_manifest(settings) or {}
    if not hist_dir.exists():
        raise FileNotFoundError(f"No existe la carga histórica en {hist_dir}; ejecute el sistema de origen.")
    checksums = {p.name: sha256_file(p) for p in sorted(hist_dir.glob("*.parquet"))}
    reg_path = lake / "_registry" / "history.json"
    reg = read_json(reg_path)
    if reg and reg.get("checksums") == checksums:
        return reg
    history_end = pd.Timestamp(manifest.get("history_end", "1900-01-02"))
    delivered = history_end - pd.Timedelta(days=1)
    summary: dict = {"checksums": checksums, "history_end": f"{history_end:%Y-%m-%d}", "sources": {}}

    known_stores = None
    known_families = None
    for source in SOURCE_ORDER:
        p = hist_dir / f"{source}.parquet"
        if not p.exists():
            continue
        raw = pd.read_parquet(p)
        ctx = _ctx(settings, granularity=_granularity(settings, source), known_stores=known_stores,
                   known_families=known_families)
        # La carga histórica es grande: se valida por filas y nunca se rechaza completa.
        ctx.max_bad_row_frac = 1.0
        res = validate_frame(raw, source, ctx)
        clean = res.clean.assign(_delivered=delivered, _version=0, _file="history")
        write_parquet(clean, lake / source / "history.parquet")
        if len(res.rejected):
            write_csv(res.rejected, quarantine / source / "history.rejected.csv")
        if source == "stores":
            known_stores = set(clean["store_nbr"].astype(int))
        if source == "sales":
            known_families = set(clean["family"].astype(str))
        summary["sources"][source] = res.summary()
        log.info("historia %s: %s", source, res.summary())
    write_json(summary, reg_path)
    return summary


# ----------------------------------------------------------------------------
# Ingesta incremental
# ----------------------------------------------------------------------------
@dataclass
class InboxEntry:
    source: str
    delivered: pd.Timestamp
    path: Path

    @property
    def stem(self) -> str:
        return self.path.stem


def list_inbox(settings: Settings, until: pd.Timestamp | None = None) -> list[InboxEntry]:
    inbox = settings.path("inbox_dir")
    out: list[InboxEntry] = []
    if not inbox.exists():
        return out
    for src_dir in sorted(p for p in inbox.iterdir() if p.is_dir()):
        for ddir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
            m = DELIVERED_RE.match(ddir.name)
            if not m:
                continue
            delivered = pd.Timestamp(m.group(1))
            if until is not None and delivered > until:
                continue
            for f in sorted(ddir.iterdir()):
                if f.is_file() and not f.name.startswith("."):
                    out.append(InboxEntry(src_dir.name, delivered, f))
    order = {s: i for i, s in enumerate(SOURCE_ORDER)}
    out.sort(key=lambda e: (e.delivered, order.get(e.source, 99), e.path.name))
    return out


def _known_masters(settings: Settings, asof: pd.Timestamp) -> tuple[set[int] | None, set[str] | None]:
    stores = load_asof(settings, "stores", asof)
    lake = settings.path("lake_dir")
    fam: set[str] | None = None
    hp = lake / "sales" / "history.parquet"
    if hp.exists():
        fam = set(pd.read_parquet(hp, columns=["family"])["family"].astype(str).unique())
    # sin historia de ventas no hay maestro de familias: no se puede usar para rechazar filas
    return (set(stores["store_nbr"].astype(int)) if len(stores) else None), (fam or None)


def read_source_csv(path: Path) -> pd.DataFrame:
    """CSV tolerante: BOM, CRLF y separador ',' o ';' (detectado en el encabezado)."""
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep, encoding="utf-8-sig")


def ingest_file(settings: Settings, entry: InboxEntry, masters_cache: dict) -> dict:
    lake = settings.path("lake_dir")
    quarantine = settings.path("quarantine_dir")
    dkey = f"delivered={entry.delivered:%Y-%m-%d}"
    reg_path = lake / "_registry" / entry.source / dkey / f"{entry.path.name}.json"
    checksum = sha256_file(entry.path)
    prev = read_json(reg_path)
    if prev and prev.get("checksum") == checksum:
        return {**prev, "skipped": True}

    record = {"source": entry.source, "file": entry.path.name, "delivered": f"{entry.delivered:%Y-%m-%d}",
              "checksum": checksum}
    qdir = quarantine / entry.source / dkey
    m = FILE_RE.match(entry.path.name)
    if entry.source not in SCHEMAS or not m or m.group("source").lower() != entry.source:
        write_json({**record, "errors": ["nombre de archivo o fuente no reconocidos"]},
                   qdir / f"{entry.path.name}.errors.json")
        record.update(status="quarantined", errors=["unrecognized_file"])
        write_json(record, reg_path)
        return record
    data_date = pd.Timestamp(m.group("date"))
    version = int(m.group("version"))
    try:
        raw = read_source_csv(entry.path)
    except Exception as exc:  # archivo ilegible
        write_json({**record, "errors": [f"ilegible: {exc}"]}, qdir / f"{entry.path.name}.errors.json")
        record.update(status="quarantined", errors=[f"unreadable: {type(exc).__name__}"])
        write_json(record, reg_path)
        return record

    if entry.delivered not in masters_cache:
        masters_cache.clear()
        masters_cache[entry.delivered] = _known_masters(settings, entry.delivered)
    stores, families = masters_cache[entry.delivered]
    ctx = _ctx(settings, data_date=None if entry.source == "stores" else data_date,
               granularity=_granularity(settings, entry.source), known_stores=stores, known_families=families)
    res = validate_frame(raw, entry.source, ctx)
    out_path = lake / entry.source / dkey / f"{entry.stem}.parquet"
    if res.ok:
        clean = res.clean.assign(_delivered=entry.delivered, _version=version, _file=entry.path.name)
        write_parquet(clean, out_path)
        if entry.source == "stores":
            masters_cache.clear()
        if len(res.rejected):
            write_csv(res.rejected, qdir / f"{entry.stem}.rejected.csv")
    else:
        write_csv(raw, qdir / entry.path.name)
        write_json({**record, **res.summary()}, qdir / f"{entry.path.name}.errors.json")
        if out_path.exists():  # reprocesamiento: un archivo que cambió y ahora es inválido sale del lago
            out_path.unlink()
    record.update(res.summary(), data_date=f"{data_date:%Y-%m-%d}", version=version)
    write_json(record, reg_path)
    return record


def ingest_until(settings: Settings, run_date: str | pd.Timestamp) -> dict:
    """Procesa todas las entregas con fecha <= run_date que aún no estén en el lago.

    Idempotente: un archivo ya procesado con el mismo checksum se omite. Si se
    ejecuta para un día del pasado (backfill) sólo considera lo entregado hasta ese día.
    """
    run_date = pd.Timestamp(run_date).normalize()
    load_history(settings)
    cache: dict = {}
    results = [ingest_file(settings, e, cache) for e in list_inbox(settings, run_date)]
    processed = [r for r in results if not r.get("skipped")]
    today = [r for r in results if r["delivered"] == f"{run_date:%Y-%m-%d}"]
    delivered_sources = sorted({r["source"] for r in today})
    expected_daily = [s for s, c in settings["source_system"]["sources"].items()
                      if c.get("granularity", "day") == "day" and int(c["lag_days"]) > 0 and s != "oil"]
    warnings = [f"sin entrega de '{s}' el {run_date:%Y-%m-%d} (posible atraso)"
                for s in expected_daily if s not in delivered_sources]
    summary = {
        "run_date": f"{run_date:%Y-%m-%d}",
        "files_seen": len(results),
        "files_processed_now": len(processed),
        "accepted_now": sum(r.get("status") == "accepted" for r in processed),
        "quarantined_now": sum(r.get("status") == "quarantined" for r in processed),
        "delivered_today": today,
        "warnings": warnings,
        "quarantined_files": [f"{r['source']}/{r['file']}" for r in processed if r.get("status") == "quarantined"],
    }
    for w in warnings:
        log.warning(w)
    for q in summary["quarantined_files"]:
        log.warning("archivo en cuarentena: %s", q)
    write_json(summary, settings.path("lake_dir") / "_runs" / f"ingest_{run_date:%Y-%m-%d}.json")
    return summary


def ingest_done(settings: Settings, run_date: str | pd.Timestamp) -> bool:
    run_date = pd.Timestamp(run_date)
    return (settings.path("lake_dir") / "_runs" / f"ingest_{run_date:%Y-%m-%d}.json").exists()


# ----------------------------------------------------------------------------
# Vistas as-of
# ----------------------------------------------------------------------------
def lake_files(settings: Settings, source: str, asof: pd.Timestamp | None) -> list[Path]:
    base = settings.path("lake_dir") / source
    if not base.exists():
        return []
    files = []
    hist = base / "history.parquet"
    if hist.exists():
        files.append(hist)
    for ddir in sorted(base.iterdir()):
        m = DELIVERED_RE.match(ddir.name)
        if m and (asof is None or pd.Timestamp(m.group(1)) <= asof):
            files.extend(sorted(ddir.glob("*.parquet")))
    return files


def _read_lake(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col in ("date", "_delivered"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    return df


def load_asof(settings: Settings, source: str, asof: pd.Timestamp | None) -> pd.DataFrame:
    """Vista de una fuente con la información disponible en ``asof`` (None = todo)."""
    schema = SCHEMAS[source]
    files = lake_files(settings, source, asof)
    if not files:
        return pd.DataFrame(columns=list(schema.columns) + META_COLS)
    df = pd.concat([_read_lake(f) for f in files], ignore_index=True)
    # La versión manda (contrato de la bandeja): una v1 que llega tarde NO pisa a una v2 ya
    # recibida. A igual versión, gana la entrega más reciente (reenvío).
    df = df.sort_values(["_version", "_delivered", "_file"], kind="stable")
    df = df.drop_duplicates(subset=list(schema.keys), keep="last")
    return df.reset_index(drop=True)


@dataclass
class Snapshot:
    asof: pd.Timestamp | None
    sales: pd.DataFrame
    promotions: pd.DataFrame
    transactions: pd.DataFrame
    oil: pd.DataFrame
    holidays: pd.DataFrame
    stores: pd.DataFrame
    files: list[str] = field(default_factory=list)

    def manifest(self, root: Path) -> list[dict]:
        out = []
        for f in self.files:
            p = Path(f)
            out.append({"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p)})
        return out


def build_snapshot(settings: Settings, asof: str | pd.Timestamp | None,
                   include_later_deliveries: bool = False) -> Snapshot:
    """Información disponible en ``asof``.

    ``include_later_deliveries=True`` (re-expresión): usa también correcciones
    entregadas después de ``asof`` pero sigue sin usar hechos con fecha >= asof.
    """
    asof_ts = None if asof is None else pd.Timestamp(asof).normalize()
    deliveries = None if include_later_deliveries else asof_ts
    data = {s: load_asof(settings, s, deliveries) for s in SOURCE_ORDER}
    files = [str(f) for s in SOURCE_ORDER for f in lake_files(settings, s, deliveries)]
    if asof_ts is not None:
        # Defensa adicional: nunca usar ventas/transacciones con fecha >= corte.
        for s in ("sales", "transactions", "oil"):
            data[s] = data[s][pd.to_datetime(data[s]["date"]) < asof_ts]
    return Snapshot(asof=asof_ts, files=files, **data)
