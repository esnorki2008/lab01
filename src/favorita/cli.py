"""CLI: ``python -m favorita <comando>``. Los DAGs llaman a las mismas funciones."""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import pandas as pd

from .config import load_settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="favorita")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("source-system", help="Construye carga histórica + bandeja desde Kaggle")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("ingest", help="Ingesta + validación de lo entregado hasta --date")
    s.add_argument("--date", required=True)
    s = sub.add_parser("weekly", help="Preparación, (re)entrenamiento, pronóstico, pedido y publicación")
    s.add_argument("--cutoff", required=True)
    s.add_argument("--restate", action="store_true", help="re-expresar con datos corregidos posteriores")
    s = sub.add_parser("train", help="Backtest + champion/challenger + entrenamiento")
    s.add_argument("--cutoff", required=True)
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("backtest", help="Compara modelos con validación temporal")
    s.add_argument("--asof", default=None, help="por defecto: fin de la carga histórica")
    s.add_argument("--out", default="reports/backtest_metrics.json")
    s = sub.add_parser("monitor", help="Seguimiento de desempeño al --date")
    s.add_argument("--date", required=True)
    s = sub.add_parser("reproduce", help="Recalcula un pronóstico publicado y verifica igualdad")
    s.add_argument("--cutoff", required=True)
    s = sub.add_parser("simulate", help="Ejecuta localmente el flujo día por día (sin Airflow)")
    s.add_argument("--start", default=None)
    s.add_argument("--end", default=None)
    s = sub.add_parser("sample-data", help="Genera datos sintéticos con el formato de Kaggle")
    s.add_argument("--out", required=True)
    s.add_argument("--days", type=int, default=420)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.cmd == "sample-data":
        from .sample_data import write_sample_kaggle
        write_sample_kaggle(Path(a.out), days=a.days)
        return 0
    settings = load_settings(a.config)
    result: object
    if a.cmd == "source-system":
        from .source_system import build_source_system
        result = build_source_system(settings, force=a.force)
    elif a.cmd == "ingest":
        from .lake import ingest_until
        result = ingest_until(settings, a.date)
    elif a.cmd == "weekly":
        from .pipeline import weekly
        result = weekly(settings, a.cutoff, restate=a.restate)
    elif a.cmd == "train":
        from .pipeline import first_cutoff
        from .training import train_and_register
        result = train_and_register(settings, a.cutoff, force=a.force, first_cutoff=first_cutoff(settings))
    elif a.cmd == "backtest":
        from .io_utils import load_manifest, write_csv, write_json
        from .lake import load_history
        from .training import backtest, summarize_backtest
        load_history(settings)
        asof = a.asof or load_manifest(settings)["history_end"]
        bt = backtest(settings, pd.Timestamp(asof))
        summary = summarize_backtest(bt)
        write_csv(bt, settings.path("reports_dir") / "backtest_folds.csv")
        result = {r["model"]: {k: v for k, v in r.items() if k != "model"}
                  for r in summary.round(5).to_dict(orient="records")}
        write_json(result, settings.root / a.out)
    elif a.cmd == "monitor":
        from .monitoring import run_monitoring
        result = run_monitoring(settings, a.date)
    elif a.cmd == "reproduce":
        from .forecasting import reproduce
        with tempfile.TemporaryDirectory() as tmp:
            result = reproduce(settings, a.cutoff, Path(tmp))
    elif a.cmd == "simulate":
        from .io_utils import load_manifest
        from .pipeline import simulate
        m = load_manifest(settings)
        simulate(settings, a.start or m["first_delivery"], a.end or m["last_delivery"])
        result = {"status": "ok"}
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
