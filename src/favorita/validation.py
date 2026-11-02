"""Validación de archivos de las fuentes.

Política (ver DECISIONES.md):

* Errores de estructura (columnas faltantes, archivo vacío o ilegible) => se
  rechaza el archivo completo (cuarentena).
* Errores de registro (fecha distinta a la del archivo, valores negativos o no
  numéricos, tienda/familia desconocida, llaves duplicadas en conflicto) => se
  rechazan esas filas; si superan ``max_bad_row_frac`` se rechaza el archivo
  completo, porque un archivo mayoritariamente malo no es confiable.
* Duplicados idénticos => se eliminan con advertencia (situación esperada).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schemas import FAMILY_KEYED, SCHEMAS, STORE_KEYED, SourceSchema

ACCEPTED = "accepted"
QUARANTINED = "quarantined"


@dataclass
class ValidationContext:
    data_date: pd.Timestamp | None = None
    granularity: str = "day"
    known_stores: set[int] | None = None
    known_families: set[str] | None = None
    max_bad_row_frac: float = 0.05
    max_sales_value: float = 1e6
    min_bad_rows: int = 3  # en archivos chicos, 1-2 filas malas no invalidan el archivo


@dataclass
class ValidationResult:
    status: str
    clean: pd.DataFrame
    rejected: pd.DataFrame
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == ACCEPTED

    def summary(self) -> dict:
        reasons = (self.rejected["_reason"].value_counts().to_dict() if len(self.rejected) else {})
        return {"status": self.status, "rows_accepted": int(len(self.clean)),
                "rows_rejected": int(len(self.rejected)), "errors": self.errors,
                "warnings": self.warnings, "reject_reasons": {str(k): int(v) for k, v in reasons.items()}}


def _empty_like(schema: SourceSchema) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in schema.columns})


def _quarantine_all(raw: pd.DataFrame, schema: SourceSchema, errors: list[str],
                    warnings: list[str]) -> ValidationResult:
    rej = raw.copy()
    rej["_reason"] = "file_rejected"
    return ValidationResult(QUARANTINED, _empty_like(schema), rej, errors, warnings)


def _to_bool(s: pd.Series) -> pd.Series:
    m = s.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
    return m


def parse_business_date(value: str, monday: bool = False) -> str:
    """Valida una fecha de negocio recibida como parámetro (YYYY-MM-DD; lunes si es un corte)."""
    d = pd.Timestamp(str(value)[:10])
    if monday and d.weekday() != 0:
        raise ValueError(f"La fecha de corte {d.date()} no es lunes")
    return f"{d:%Y-%m-%d}"


def validate_frame(raw: pd.DataFrame, source: str, ctx: ValidationContext) -> ValidationResult:
    schema = SCHEMAS[source]
    errors: list[str] = []
    warnings: list[str] = []
    raw = raw.copy()
    raw.columns = [str(c).strip().lstrip("﻿").lower() for c in raw.columns]

    missing = [c for c in schema.columns if c not in raw.columns]
    if missing:
        return _quarantine_all(raw, schema, [f"columnas faltantes: {missing}"], warnings)
    extra = [c for c in raw.columns if c not in schema.columns]
    if extra:
        warnings.append(f"columnas extra ignoradas: {extra}")
    df = raw[list(schema.columns)].copy()
    if len(df) == 0:
        if schema.allow_empty:
            return ValidationResult(ACCEPTED, _empty_like(schema), df.assign(_reason=[]), errors, ["archivo vacío"])
        return _quarantine_all(raw, schema, ["archivo vacío"], warnings)

    reason = pd.Series("", index=df.index, dtype=object)

    def flag(mask: pd.Series, why: str) -> None:
        mask = mask.fillna(False) & (reason == "")
        reason.loc[mask] = why

    # --- llaves no nulas ---------------------------------------------------
    for k in schema.keys:
        s = df[k].astype(str).str.strip()
        flag(df[k].isna() | (s == "") | (s.str.lower() == "nan"), f"null_{k}")

    # --- fecha -------------------------------------------------------------
    if schema.has_date:
        # tolera "2017-08-01 00:00:00": sólo importa la parte de fecha
        d = pd.to_datetime(df["date"].astype(str).str.strip().str[:10], errors="coerce", format="%Y-%m-%d")
        flag(d.isna(), "invalid_date")
        if ctx.data_date is not None:
            if ctx.granularity == "month":
                period = d.dt.to_period("M").dt.to_timestamp()
                flag(period != ctx.data_date, "date_mismatch")
            else:
                flag(d != ctx.data_date, "date_mismatch")
        df["date"] = d

    # --- tipos -------------------------------------------------------------
    for col in schema.numeric + tuple(c for c in schema.integer if c not in schema.numeric):
        txt = df[col].astype(str).str.strip()
        empty = df[col].isna() | (txt == "") | (txt.str.lower() == "nan")
        num = pd.to_numeric(txt.where(~empty), errors="coerce")
        if col in schema.nullable:
            flag(~empty & num.isna(), f"non_numeric_{col}")
        else:
            flag(num.isna(), f"non_numeric_{col}")
        flag(~np.isfinite(num.fillna(0.0)), f"non_finite_{col}")
        if col in schema.integer:
            flag(num.notna() & (num != np.round(num)), f"non_integer_{col}")
        if col in schema.non_negative:
            flag(num < 0, f"negative_{col}")
        df[col] = num
    if source == "sales":
        flag(df["sales"] > ctx.max_sales_value, "sales_out_of_range")
    for col, domain in schema.categorical_domains.items():
        flag(~df[col].astype(str).isin(domain), f"unknown_{col}")
    if source == "holidays":
        tb = _to_bool(df["transferred"])
        flag(tb.isna(), "invalid_transferred")
        df["transferred"] = tb
    for col in ("family", "city", "state", "type", "locale_name", "description"):
        if col in df.columns and col not in schema.categorical_domains:
            df[col] = df[col].astype(str).str.strip()
    if "family" in df.columns:
        df["family"] = df["family"].str.upper()

    # --- integridad referencial con maestros -------------------------------
    if source in STORE_KEYED and ctx.known_stores is not None:
        flag(~df["store_nbr"].isin(ctx.known_stores), "unknown_store")
    if source in FAMILY_KEYED and ctx.known_families is not None:
        flag(~df["family"].isin(ctx.known_families), "unknown_family")

    good = reason == ""
    clean = df.loc[good].copy()

    # --- duplicados --------------------------------------------------------
    n_before = len(clean)
    clean = clean.drop_duplicates()
    if len(clean) < n_before:
        warnings.append(f"{n_before - len(clean)} filas duplicadas idénticas eliminadas")
    conflict = clean.duplicated(subset=list(schema.keys), keep=False)
    if conflict.any():
        idx = clean.index[conflict]
        reason.loc[idx] = "conflicting_duplicate"
        clean = clean.loc[~conflict]

    rejected = raw.loc[reason != ""].copy()
    rejected["_reason"] = reason[reason != ""]
    # Las filas de tiendas/familias aún no dadas de alta son un problema de maestros, no evidencia de
    # un archivo corrupto: no cuentan para rechazar el archivo completo (si no, se perderían las
    # filas válidas, p. ej. una corrección). Se rechazan individualmente.
    corrupt = ~rejected["_reason"].isin(["unknown_store", "unknown_family"])
    n_bad = int(corrupt.sum())
    bad_frac = n_bad / max(len(df), 1)
    all_bad = n_bad == len(df)
    if all_bad or (bad_frac > ctx.max_bad_row_frac and n_bad >= ctx.min_bad_rows):
        errors.append(f"{bad_frac:.1%} de filas inválidas (> {ctx.max_bad_row_frac:.0%})")
        return _quarantine_all(raw.assign(_row_reason=reason), schema, errors, warnings)
    if len(rejected):
        warnings.append(f"{len(rejected)} filas rechazadas")

    # tipos finales
    for col in schema.integer:
        clean[col] = clean[col].astype("int64")
    for col in schema.numeric:
        if col not in schema.integer:
            clean[col] = clean[col].astype("float64")
    return ValidationResult(ACCEPTED, clean.reset_index(drop=True), rejected, errors, warnings)
