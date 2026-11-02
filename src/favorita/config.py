"""Carga de configuración y resolución de rutas del proyecto."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = "config/params.yaml"


def project_root() -> Path:
    env = os.environ.get("FAVORITA_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    root: Path
    raw: dict[str, Any]
    config_path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, key: str) -> Path:
        return (self.root / self.raw["paths"][key]).resolve()

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    # --- atajos de negocio -------------------------------------------------
    def family_costs(self, family: str) -> tuple[float, float]:
        b = self.raw["business"]
        overrides = b.get("family_overrides") or {}
        if family in overrides:
            o = overrides[family]
            return float(o["underage_cost"]), float(o["overage_cost"])
        for group in (b.get("cost_groups") or {}).values():
            if family in group.get("families", []):
                return float(group["underage_cost"]), float(group["overage_cost"])
        d = b["default_costs"]
        return float(d["underage_cost"]), float(d["overage_cost"])


def load_settings(config_path: str | Path | None = None, root: str | Path | None = None,
                  overrides: dict[str, Any] | None = None) -> Settings:
    root_p = Path(root).resolve() if root else project_root()
    cfg = Path(config_path) if config_path else Path(os.environ.get("FAVORITA_CONFIG", DEFAULT_CONFIG))
    if not cfg.is_absolute():
        cfg = root_p / cfg
    with open(cfg, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if overrides:
        raw = deep_merge(raw, overrides)
    return Settings(root=root_p, raw=raw, config_path=cfg)


def deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
