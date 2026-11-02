"""Contrato de datos de cada fuente: columnas, tipos y llaves."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceSchema:
    name: str
    columns: tuple[str, ...]
    keys: tuple[str, ...]
    numeric: tuple[str, ...] = ()
    integer: tuple[str, ...] = ()
    non_negative: tuple[str, ...] = ()
    nullable: tuple[str, ...] = ()
    has_date: bool = True
    allow_empty: bool = False
    categorical_domains: dict[str, tuple[str, ...]] = field(default_factory=dict)


HOLIDAY_TYPES = ("Holiday", "Transfer", "Additional", "Bridge", "Work Day", "Event")
HOLIDAY_LOCALES = ("Local", "Regional", "National")

SCHEMAS: dict[str, SourceSchema] = {
    "sales": SourceSchema(
        name="sales",
        columns=("date", "store_nbr", "family", "sales"),
        keys=("date", "store_nbr", "family"),
        numeric=("sales",), integer=("store_nbr",), non_negative=("sales",),
    ),
    "promotions": SourceSchema(
        name="promotions",
        columns=("date", "store_nbr", "family", "onpromotion"),
        keys=("date", "store_nbr", "family"),
        numeric=("onpromotion",), integer=("store_nbr", "onpromotion"), non_negative=("onpromotion",),
    ),
    "transactions": SourceSchema(
        name="transactions",
        columns=("date", "store_nbr", "transactions"),
        keys=("date", "store_nbr"),
        numeric=("transactions",), integer=("store_nbr", "transactions"), non_negative=("transactions",),
    ),
    "oil": SourceSchema(
        name="oil",
        columns=("date", "dcoilwtico"),
        keys=("date",),
        numeric=("dcoilwtico",), non_negative=("dcoilwtico",), nullable=("dcoilwtico",),
    ),
    "holidays": SourceSchema(
        name="holidays",
        columns=("date", "type", "locale", "locale_name", "description", "transferred"),
        keys=("date", "type", "locale", "locale_name", "description"),
        allow_empty=True,
        categorical_domains={"type": HOLIDAY_TYPES, "locale": HOLIDAY_LOCALES},
    ),
    "stores": SourceSchema(
        name="stores",
        columns=("store_nbr", "city", "state", "type", "cluster"),
        keys=("store_nbr",),
        numeric=("cluster",), integer=("store_nbr", "cluster"),
        has_date=False,
    ),
}

# Fuentes cuya validación depende de los maestros (tiendas / familias conocidas)
STORE_KEYED = ("sales", "promotions", "transactions")
FAMILY_KEYED = ("sales", "promotions")
