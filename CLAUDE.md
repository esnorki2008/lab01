# CLAUDE.md — reglas del proyecto para agentes de programación

Producto: pronóstico semanal de ventas (cuantiles) y pedido sugerido por tienda × familia
para Corporación Favorita. Orquestado con Airflow 3, datos versionados con DVC.

## Estructura
- `src/favorita/` — TODA la lógica (validación, lago as-of, features, modelos, pedido, monitoreo).
- `dags/` — DAGs de Airflow. **Sólo orquestan**: cada tarea importa una función de `favorita` y la llama.
  Prohibido importar pandas/numpy/lightgbm a nivel de módulo o leer datos en un DAG (lo verifica `tests/test_dags.py`).
- `config/params.yaml` — única fuente de parámetros (costos, horizonte, cuantiles, criterio de reemplazo).
  Un cambio de negocio se hace aquí, nunca en el código.
- `tests/` — pytest; datos sintéticos con formato Kaggle (`favorita.sample_data`).
- `DECISIONES.md` — registrar toda decisión técnica/de producto nueva con alternativas y justificación.

## Reglas no negociables
1. **Disponibilidad en el tiempo**: un pronóstico con corte T sólo puede usar `lake.build_snapshot(settings, T)`
   (entregas con `delivered <= T`, hechos con fecha `< T`). Nunca leer la bandeja o Kaggle directamente para modelar.
2. **La bandeja (`data/inbox`) es un sistema externo**: el producto la lee, nunca la modifica.
   Sólo `source_system.py` (que simula a la empresa) escribe ahí.
3. **Idempotencia**: toda tarea debe poder re-ejecutarse con el mismo resultado. Escribir con
   `io_utils.write_*` (escritura atómica), rutas determinísticas por fecha, sin timestamps de reloj en los datos.
4. **Lo publicado es inmutable**: `data/forecasts/cutoff=T` no se sobrescribe con contenido distinto;
   las re-expresiones van a `data/forecasts/restated/`.
5. **Datos inválidos se detienen en la ingesta** (cuarentena). Situaciones esperadas (atrasos, entregas parciales,
   correcciones, feriados, sin datos de petróleo en fin de semana) **no deben hacer fallar** el flujo.
6. **Nada de datos ni modelos en Git**: se versionan con DVC (`dvc add` / `dvc push`). Sólo punteros `.dvc`.
7. Trabajar en ramas y abrir Pull Request a `main` (protegida); CI (`ruff` + `pytest`) debe pasar.

## Comandos
```bash
pip install -r requirements-dev.txt
pytest -q                     # pruebas (rápidas, datos sintéticos)
ruff check src dags tests     # lint
PYTHONPATH=src python -m favorita --help   # CLI: source-system, ingest, weekly, train, monitor, reproduce, simulate
docker compose up -d --build  # Airflow en http://localhost:8080
```

## Convenciones
- Python 3.11, líneas ≤ 120, tipos en firmas públicas, docstrings en español.
- Fechas de corte = lunes (`forecast.cutoff_weekday`). Columnas de cuantiles: `q10`, `q25`, `q50`, ...
- Nuevas fuentes: declarar esquema en `schemas.py`, regla de llegada en `params.yaml: source_system.sources`.
