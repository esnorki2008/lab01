# Favorita — Pronóstico de ventas y pedido sugerido

Producto de IA que, cada semana y sin intervención manual, transforma los datos de ventas que llegan
diariamente en:

1. **Pronóstico diario probabilístico** (cuantiles 10/25/50/75/90) a 14 días por tienda × familia.
2. **Pedido sugerido** al centro de distribución, derivado del pronóstico y de los costos de quiebre y de
   merma definidos en `config/params.yaml` (newsvendor).
3. **Seguimiento del desempeño**: pronósticos publicados vs. ventas reales a medida que se conocen
   (incluidas las correcciones).

Todo resultado publicado se puede rastrear al commit de código, a los archivos de datos (con su hash)
y al modelo que lo produjo, y se puede **reproducir bit a bit** (`python -m favorita reproduce`).

> Decisiones técnicas y de producto, alternativas y justificación: **[DECISIONES.md](DECISIONES.md)**.
> Reglas para agentes de programación: **[CLAUDE.md](CLAUDE.md)**.

---

## 1. Requisitos

- Docker Desktop / Docker Engine con Compose v2 y **≥ 8 GB de RAM** asignados a Docker.
- Git. (Opcional, para trabajar fuera de Docker: Python 3.11.)
- ~5 GB de disco libres.

## 2. Datos (Kaggle)

Descargue de <https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data> y coloque
**sin renombrar** en esta ruta:

```
data/store-sales-time-series-forecasting/
├── train.csv
├── stores.csv
├── holidays_events.csv
├── oil.csv
└── transactions.csv          (test.csv y sample_submission.csv no se usan)
```

Con la CLI de Kaggle: `kaggle competitions download -c store-sales-time-series-forecasting -p data/store-sales-time-series-forecasting && unzip ...`

## 3. Levantar y ejecutar desde cero

```bash
git clone <repo> && cd lab01
# (colocar los archivos de Kaggle, sección 2)
# Linux: que los archivos creados por Airflow sean de su usuario
echo "AIRFLOW_UID=$(id -u)" > .env          # en Windows/macOS no es necesario
mkdir -p ../dvc-storage                     # remoto DVC local (si no existe, Docker lo crea como root)
docker compose up -d --build
```

UI de Airflow: <http://localhost:8080> (acceso sin contraseña, modo laboratorio).

**No hay que hacer nada más.** Los DAGs se crean des-pausados y la secuencia ocurre sola:

| DAG | Calendario | Qué hace |
|---|---|---|
| `favorita_source_system` | una vez | *Simula a la empresa*: con los archivos de Kaggle crea la **carga histórica** (`data/history`) y la **bandeja** (`data/inbox`), un archivo por fecha y fuente, con atrasos, entregas parciales, correcciones y archivos inválidos. |
| `favorita_ingest` | diario | Carga histórica inicial (una vez) + ingesta y **validación** de lo entregado hasta la fecha. Lo válido va al lago (`data/lake`), lo inválido a cuarentena (`data/quarantine`). |
| `favorita_weekly_forecast` | lunes | Espera la ingesta → prepara features (*as-of*) → reentrena/selecciona modelo cada 4 semanas → pronostica → pedido sugerido → **publica** → `dvc add`/`dvc push`. |
| `favorita_monitoring` | diario | Compara todos los pronósticos publicados con las ventas reales conocidas a la fecha. |
| `favorita_reprocess` | manual | Re-expresa un periodo con datos corregidos (sin tocar lo publicado). |

Cuando `favorita_source_system` termina, el scheduler vuelve a leer los DAGs (≈30 s) y éstos quedan con
calendario desde el primer día de la bandeja hasta la última entrega; con `catchup` Airflow recorre
todo el periodo automáticamente (≈ 26 semanas; 1–2 h según la máquina, cada reentrenamiento ≈ 3 min).

### Resultados (todo bajo `data/`, versionado con DVC)

| Ruta | Contenido |
|---|---|
| `data/forecasts/cutoff=YYYY-MM-DD/forecast.parquet` | pronóstico por día, tienda, familia y cuantil |
| `data/forecasts/cutoff=YYYY-MM-DD/orders.csv` | **pedido sugerido** (cantidad, nivel de servicio, costos, comparación con pedido ingenuo) |
| `data/forecasts/cutoff=YYYY-MM-DD/metadata.json` | commit de código, hash de config, modelo (ruta + sha256), hash de resultados, chequeos de disponibilidad |
| `data/forecasts/cutoff=YYYY-MM-DD/inputs_manifest.json` | cada archivo del lago usado, con su sha256 |
| `data/forecasts/restated/...` | re-expresiones con datos corregidos (nunca reemplazan lo oficial) |
| `data/monitoring/metrics_by_cutoff.csv` | WAPE, sesgo, pinball, cobertura 80%, calibración de cada cuantil (`cal_q10`…`cal_q90`), costo del pedido vs. pedido ingenuo, por corte |
| `reports/monitoring_report.md` | reporte legible del seguimiento con alertas |
| `models/registry.json` | historial champion/challenger: backtest de cada candidato y decisión |
| `data/quarantine/` | archivos/filas rechazados y el motivo (`*.errors.json`) |
| `data/lake/_runs/ingest_*.json` | resumen de cada ingesta (aceptados, rechazados, atrasos) |
| `reports/source_system_anomalies.csv` | "verdad" de las anomalías inyectadas por el simulador (para auditar) |

## 4. Periodo de evaluación / datos no publicados

El periodo operado se deriva **de los datos**, no del código:

- Por defecto (`source_system.history_end: auto`) la bandeja cubre las últimas `inbox_days` (182) días
  del `train.csv` presente, alineado a lunes; todo lo anterior es la carga histórica. Si se coloca un
  `train.csv` con un periodo distinto, el producto opera automáticamente sobre ese periodo.
- Para fijar el inicio: `source_system.history_end: "2017-05-01"` en `config/params.yaml`.
- También se pueden **dejar archivos directamente en la bandeja** con el contrato de la sección 6; el
  calendario de los DAGs se extiende solo hasta la última carpeta `delivered=` encontrada. Si la
  bandeja no la creó el simulador, no se modifica: sólo se construye la carga histórica desde Kaggle
  hasta el día anterior a la primera entrega.
- Salvaguardas: se ignoran fechas aisladas mal escritas y se garantizan ≥ 150 días de historia.

## 5. Operación: backfill, reprocesamiento, reproducción

Los comandos se ejecutan dentro del contenedor (`docker compose exec airflow-scheduler <cmd>`):

```bash
# Backfill (Airflow 3) de un periodo pasado; las tareas son idempotentes
airflow backfill create --dag-id favorita_ingest --from-date 2017-03-01 --to-date 2017-03-31 --reprocess-behavior completed
airflow backfill create --dag-id favorita_weekly_forecast --from-date 2017-03-06 --to-date 2017-03-27 --reprocess-behavior completed

# Reprocesar un periodo con las correcciones llegadas después (resultado en data/forecasts/restated)
airflow dags trigger favorita_reprocess --conf '{"start_date": "2017-03-06", "end_date": "2017-04-03"}'

# Re-expresar un solo corte (los disparos manuales SIEMPRE requieren run_date, que debe ser lunes)
airflow dags trigger favorita_weekly_forecast --conf '{"run_date": "2017-03-06", "restate": true}'

# Verificar que un pronóstico publicado se reproduce idéntico
python -m favorita reproduce --cutoff 2017-03-06
```

**Reproducir un pronóstico del pasado desde cero** (otra máquina):

```bash
git log -- data/forecasts.dvc               # elegir el commit en que se publicó
git checkout <commit>
dvc pull                                    # trae lago, modelos y pronósticos de esa versión
python -m favorita reproduce --cutoff 2017-03-06   # recalcula y compara hashes -> "identical": true
```

Como el lago es *append-only* y el pronóstico usa la vista *as-of* del corte, cualquier versión posterior
de los datos también reproduce los pronósticos antiguos.

## 6. Contrato de la bandeja (sistema de origen)

```
data/inbox/<fuente>/delivered=<fecha de entrega>/<fuente>_<fecha del dato>_v<versión>.csv
```

| Fuente | Columnas | Llegada (config) |
|---|---|---|
| `sales` | date, store_nbr, family, sales | diaria, D+1 |
| `transactions` | date, store_nbr, transactions | diaria, D+2 |
| `oil` | date, dcoilwtico | días hábiles, D+1 |
| `promotions` | date, store_nbr, family, onpromotion | plan, 14 días antes |
| `holidays` | date, type, locale, locale_name, description, transferred | mensual, 30 días antes |
| `stores` | store_nbr, city, state, type, cluster | maestro (opcional) |

Una versión mayor del mismo dato **reemplaza fila a fila** (por llave) a las anteriores desde el día en que
llega: sirve tanto para entregas por partes como para correcciones.

## 7. DVC

El repositorio ya trae `dvc init` y un remoto local `../dvc-storage` (carpeta hermana del repo, montada
en el contenedor). El DAG semanal ejecuta `dvc add` + `dvc push` al publicar. Para fijar la versión en Git:

```bash
dvc add data/store-sales-time-series-forecasting   # datos de entrada (una vez)
git add data/*.dvc models.dvc dvc.lock && git commit -m "data: snapshot <fecha>"
dvc push
```

Pipeline reproducible fuera de Airflow: `dvc repro` (simulación de la bandeja + backtest de modelos,
métricas en `reports/backtest_metrics.json`, ver `dvc metrics show`).
Para usar otro remoto: `dvc remote add -d -f storage s3://...` (o gdrive, etc.).

## 8. Desarrollo local (sin Docker)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q
export PYTHONPATH=src            # Windows PowerShell: $env:PYTHONPATH="src"
python -m favorita source-system
python -m favorita simulate      # mismo flujo que Airflow, día por día
```

## 9. Flujo de trabajo Git

- `main` protegida: requiere Pull Request y que pase el check **CI** (`.github/workflows/ci.yml`: ruff,
  pytest y carga de DAGs en Airflow 3). Configuración: *Settings → Branches → Add rule → main →
  Require a pull request before merging + Require status checks to pass (CI)*, o con `gh`:
  ```bash
  gh api -X PUT repos/<owner>/lab01/branches/main/protection --input .github/branch-protection.json
  ```
- Ramas `feature/*`, `fix/*`, `data/*`; PR con la plantilla `.github/pull_request_template.md`.

## 10. Estructura

```
config/params.yaml        parámetros de negocio, datos, modelo y criterio de reemplazo (versionado)
src/favorita/             lógica: source_system, validation, lake, features, models, ordering,
                          metrics, training, forecasting, monitoring, versioning, cli
dags/                     DAGs de Airflow (sólo orquestación)
tests/                    pytest (unitarias + punta a punta con datos sintéticos)
docker/Dockerfile         imagen Airflow 3.1 + dependencias + DVC
docker-compose.yaml       Airflow (api-server, scheduler, dag-processor, triggerer) + PostgreSQL
dvc.yaml                  pipeline DVC (bandeja + backtest)
```
