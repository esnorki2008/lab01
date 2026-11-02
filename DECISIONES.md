# DECISIONES.md

Registro de decisiones técnicas y de producto: qué se decidió, qué alternativas había y por qué.
Los parámetros mencionados viven en `config/params.yaml`.

---

## 1. Definición del producto

| Decisión | Alternativas | Justificación |
|---|---|---|
| **Granularidad tienda × familia, diaria**; corte semanal los **lunes**. | SKU (no existe en los datos); corte diario. | Es la granularidad de los datos y la del pedido al CD. Pedido semanal = proceso actual del negocio. |
| **Horizonte 14 días** (T … T+13). | 7 o 16 días. | Cubre la ventana del pedido (lead time 2 + revisión 7 = hasta T+8) con margen para planificar el siguiente; más horizonte degrada la calidad sin uso de negocio. |
| **Pronóstico en cuantiles** 0.10, 0.25, 0.50, 0.75, 0.90. | Pronóstico puntual + error normal; distribución paramétrica. | El pedido necesita un cuantil que depende del costo; con 5 cuantiles se interpola cualquier nivel de servicio 10–90 % sin reentrenar. Una normal no representa bien la asimetría ni los ceros. |
| **Pedido = cantidad a recibir** para cubrir la ventana `[T+lead_time, T+lead_time+review−1]`. | Pedido neto de inventario. | No hay datos de inventario. Se documenta que el pedido real = sugerido − inventario disponible; agregar una fuente de inventario no cambia la lógica (resta). |

## 2. Sistema de origen simulado (bandeja)

- **Carga histórica + bandeja con las mismas reglas.** A cada registro de Kaggle se le asigna una *fecha
  de entrega* = fecha del dato + `lag_days` de su fuente. La carga histórica es exactamente lo que ya se
  había entregado antes de `history_end`; la bandeja es todo lo que se entrega después. Así no hay un
  "corte artificial" distinto entre historia y operación.
- **Frecuencias y anticipaciones distintas por fuente** (configurables):
  ventas D+1, transacciones D+2, petróleo D+1 sólo días hábiles, plan de promociones **14 días antes**,
  feriados por mes **30 días antes**. Las promociones y feriados son información de *futuro* que el
  negocio sí conoce al pedir; por eso son legítimas como features del día objetivo.
- **Situaciones de operación real inyectadas** (determinísticas por semilla, auditables en
  `reports/source_system_anomalies.csv`): entrega de ventas **en dos partes** (≈30 % de tiendas llega
  un día después), **correcciones** de valores (×0 o ×10 en 2 % de filas, corregidas 5 días después),
  **archivos inválidos** (valores negativos, texto en columnas numéricas, columna faltante, fecha
  equivocada) reenviados correctos 2 días después, **atrasos** de 1–3 días y **filas duplicadas**.
- **`history_end: auto`**: la bandeja cubre los últimos 182 días del `train.csv` presente (alineado a
  lunes). Alternativa: fecha fija. Se eligió `auto` para que, si se colocan datos de otro periodo (p. ej.
  un periodo no publicado), el producto opere sobre él sin tocar código ni configuración.
  Salvaguardas: (a) el rango operativo se calcula con fechas que traen ventas de ≥ 50 % de las tiendas,
  así una fecha mal escrita (p. ej. 2071-08-10) no traslada la operación; (b) se garantizan ≥ 150 días
  de historia: con un `train.csv` corto se acorta la bandeja en vez de dejar al modelo sin datos.
- **Bandeja externa**: si `data/inbox` ya tiene entregas que no generó el simulador, no se toca; sólo se
  construye la carga histórica desde Kaggle hasta el día previo a la primera entrega, y el manifiesto
  se guarda junto a la historia (`data/history/_manifest.json`).
- **La bandeja es externa**: el producto sólo la lee (verificado por prueba: checksums antes/después).
  El simulador es un DAG aparte que representa a la empresa y no se ejecuta si la bandeja no es suya.

## 3. Disponibilidad de la información en el tiempo (principio fundamental)

- **Dos fechas por registro**: fecha del dato y fecha de entrega (`_delivered`). El lago guarda cada
  archivo aceptado bajo `delivered=YYYY-MM-DD`, *append-only*.
- **Vista as-of**: el pronóstico con corte T usa `build_snapshot(T)` = carga histórica + entregas con
  `delivered ≤ T`, y además descarta hechos con fecha ≥ T (defensa en profundidad). Para cada llave se
  toma el valor de la entrega más reciente *hasta T*. Si el domingo no llegó (atraso), el lunes se
  pronostica con lo que hay: los features toleran faltantes (`NaN` ≠ 0).
- **Features "conocibles"**: medias móviles, rezagos del mismo día de semana, año anterior, etc. se
  calculan sólo con ventas < T; transacciones con fecha < T − 1 (su lag); petróleo último valor < T; del
  día objetivo sólo calendario, plan de promociones y feriados (conocidos por adelantado).
- **Datos faltantes ≠ cero.** Si el último día no llegó (atraso) o llegó sólo una parte de las tiendas,
  el lunes falta `T−1`. Primera versión: el modelo sólo había visto faltantes en series muertas y
  pronosticaba ~0 (corte 2017-05-15: WAPE 0.248). Solución: (1) imputar `last_1` con la media de la
  semana y los rezagos del mismo día con su media de 8 semanas, con banderas `*_missing`; (2) **entrenar
  con faltantes simulados** en `T−1` (10 % de los cortes sin el día completo, 30 % sin ~30 % de las
  tiendas), igual que en producción. Resultado en ese corte: WAPE 0.125 y −54 % de costo del pedido.
- **Ceros antes de la primera venta** (tienda o producto que aún no existía) se tratan como
  desconocidos, no como demanda 0, para no subestimar tiendas nuevas.
- **Misma función para entrenar y predecir** (`features.cutoff_features`): no hay divergencia
  entrenamiento/producción. Prueba `test_features_do_not_use_information_after_cutoff`: alterar ventas
  ≥ T no cambia ningún feature.
- **Trade-off aceptado**: al entrenar en el corte T, las filas de cortes pasados c < T se construyen con la
  vista as-of T (que incluye correcciones posteriores a c), no con la vista as-of c. Alternativa: un
  snapshot por cada corte de entrenamiento (≈ 30× más lectura). Se acepta porque las correcciones son
  pocas y el modelo aprende de los datos "verdaderos"; lo que nunca ocurre es usar información de fechas
  ≥ T. El backtest usa la misma convención para todos los modelos, así que la comparación es justa.

## 4. Validación de datos

| Nivel | Regla | Acción |
|---|---|---|
| Archivo | nombre/fuente no reconocidos, ilegible, columnas faltantes, vacío (excepto feriados) | **cuarentena del archivo completo** |
| Fila | fecha ilegible o distinta a la del archivo, llave nula, no numérico/no entero/infinito, negativo, venta > 1e6, tienda o familia desconocida, `transferred` inválido, dominio de `type`/`locale` | fila rechazada (a `*.rejected.csv`) |
| Archivo | **> 5 % de filas inválidas** (`max_bad_row_frac`) | cuarentena del archivo completo |
| Duplicados | idénticos → se eliminan (advertencia); misma llave con valores distintos → ambas filas rechazadas | — |
| Formato | BOM, CRLF, separador `;`, encabezados en mayúsculas, fecha con hora, familia en minúsculas, extensión `.CSV` | se normalizan (no son errores de datos) |

- **Integridad referencial no cuenta para el umbral**: filas de tiendas/familias aún no dadas de alta se
  rechazan individualmente pero no hacen rechazar el archivo, para no perder las filas válidas (p. ej.
  una corrección). En archivos chicos se exigen ≥ 3 filas malas para rechazar el archivo completo.
- **Valores extremos que pasan la validación** (p. ej. un ×10 mal digitado) se acotan *en los features*
  a 4× el percentil 95 de los últimos 112 días de la serie; el valor real se conserva.

- **Por qué combinar archivo y fila**: rechazar un archivo de 1 782 filas por una fila mala bloquea al
  negocio; aceptar un archivo con 30 % de negativos significa que el archivo entero no es confiable.
- **Maestros as-of**: la tienda debe existir en el maestro vigente a la fecha de entrega (permite altas
  futuras vía fuente `stores`); familias, las conocidas en la carga histórica.
- **Esperado vs. incidente**: atrasos, entregas parciales, correcciones, días sin petróleo, feriados,
  tiendas cerradas (ventas 0) y archivos en cuarentena **no fallan** el flujo (se registran en
  `data/lake/_runs/` y como *warnings*). Un incidente sí detiene la publicación (`DataQualityError`):
  ventas con **> 7 días de atraso** o **< 50 %** de las series activas con datos en la última semana.
  Publicar un pedido con datos tan viejos es peor que no publicarlo.
- La carga histórica se valida igual pero sólo por fila (nunca se rechaza completa).

## 5. Manejo de correcciones

- **Upsert por llave según versión**: una versión posterior (`_v2`, `_v3`…) del mismo dato reemplaza
  *fila a fila* a las anteriores desde el día en que llega. Manda el número de versión (contrato de la
  bandeja): una v1 que llega tarde **no** pisa a una v2 ya recibida; a igual versión gana la entrega
  más reciente (reenvío). Funciona igual para "llega por partes"
  (la parte 2 trae otras llaves) y para correcciones (trae sólo las llaves corregidas).
  Alternativas: reemplazo del archivo completo (rompe las entregas por partes) o sobrescribir el lago
  (pierde la historia y la reproducibilidad). Limitación: no se modelan borrados.
- **Lo publicado es inmutable**: un pronóstico oficial no se recalcula cuando llega una corrección,
  porque refleja la información disponible al decidir. Si se re-ejecuta un corte y el resultado difiere
  (código nuevo o datos), el original se conserva y el nuevo va a `data/forecasts/restated/`.
- **Reprocesamiento**: (a) el seguimiento se recalcula completo en cada corrida con las ventas as-of
  la fecha de ejecución → refleja correcciones automáticamente (`actual_rows_v2plus` en el reporte);
  (b) los cortes posteriores ya usan los datos corregidos; (c) `favorita_reprocess` o `restate=true`
  re-expresan cortes pasados con las correcciones para análisis.

## 6. Idempotencia, backfill y orquestación

- **Ingesta "barrido hasta D"**: la corrida del día D procesa todo lo entregado ≤ D que no esté en el
  registro (`_registry/…json` con checksum). Re-ejecutar D no hace nada; correr un backfill desde
  cualquier fecha no deja huecos aunque las corridas anteriores no hayan existido.
- **Escrituras atómicas** (archivo temporal + `os.replace`) y rutas determinísticas por fecha. Ningún
  dato depende del reloj. LightGBM con `deterministic=True` y semilla fija → mismo modelo, mismo hash.
- **Publicación**: si existe un pronóstico con el mismo hash → *no-op*.
- **Monitoreo** usa cortes estrictamente anteriores a la fecha de ejecución, para que su resultado no
  dependa de si el DAG semanal del mismo día ya terminó.
- **Registro de modelos indexado por fecha de corte**: el modelo de un corte es la última decisión con
  fecha ≤ corte; un backfill de un corte viejo usa el modelo vigente en ese momento.
- **Dependencias entre DAGs**: sensores (`mode=reschedule`) sobre el marcador `ingest_D.json`, en lugar
  de `ExternalTaskSensor`, para no acoplarse a la semántica de fechas lógicas entre DAGs.
- **Calendario derivado de los datos**: los DAGs leen el manifiesto de la bandeja (JSON pequeño) para su
  `start_date`/`end_date`; se extiende si aparecen entregas posteriores. `catchup=True` +
  `max_active_runs=1` recorre el periodo en orden sin intervención. Alternativa descartada: fechas fijas en
  el código (no sirve para un periodo de evaluación desconocido).
- **Disparos manuales**: sin `run_date` fallan de inmediato con un mensaje claro (antes tomaban la fecha
  de hoy y el sensor esperaba 24 h); las fechas de corte se validan como lunes.
- **Cierres de cadena** (p. ej. 25-dic): se infieren de la historia (días sin ventas en toda la
  cadena); el pronóstico de ese día es 0 y el monitoreo registra la venta real como 0.
- **Los DAGs sólo orquestan** (prueba AST: sin pandas/numpy a nivel de módulo, siempre delegan a
  `favorita.*`).

## 7. Modelado

- **Baseline**: cuantiles empíricos de las últimas 8 ventas del mismo día de la semana (naive estacional
  probabilístico). Sin entrenamiento, robusto, fácil de explicar; es el champion inicial.
- **Modelo de ML**: **LightGBM global** (todas las series), un modelo por cuantil con objetivo
  `quantile`, sobre `log1p(ventas)` (los cuantiles son invariantes a transformaciones monótonas).
  Estrategia **directa multi-horizonte** con `h` como feature (sin recursión, sin acumular error).
  ≈ 34 features: nivel/tendencia/volatilidad recientes, rezagos del mismo día de semana, año anterior,
  promociones planificadas, feriados nacional/regional/local, transacciones de la tienda, petróleo,
  calendario (día de pago 15 y fin de mes), atributos de tienda y familia.
  Ventana de entrenamiento 224 días de cortes semanales (≈ 0.9 M filas); ≈ 30 s por ajuste.
  Se probaron variantes (objetivo L1, residual sobre el nivel, Tweedie, árboles más grandes): ninguna
  mejoró de forma consistente.
- **Deep learning: no se usa.** Con ~1 800 series y ~4 años, los modelos de árboles globales igualan o
  superan a redes (N-BEATS/TFT) en esta competencia, con una fracción del costo: entrenan en segundos en
  CPU dentro del contenedor de Airflow, sin GPU, y son deterministas (clave para reproducibilidad).
  El beneficio esperado no justifica el costo operativo.
- **Validación temporal (rolling origin)**: 4 cortes semanales consecutivos que terminan 14 días antes
  de T; cada modelo se entrena sólo con objetivos < corte del fold y se evalúa en sus 14 días.
  **Mismos cortes, mismos datos y mismas métricas para todos los candidatos.** Los features de cada
  fold se construyen con la vista *as-of del fold* (con sus atrasos y entregas parciales), no con la
  historia ya completa: así el backtest mide el desempeño en las condiciones reales de operación.
- **Métricas técnicas**: WAPE (mediana), sesgo, RMSLE (la de Kaggle), pinball escalada y cobertura del
  intervalo 10–90 % (objetivo 0.80).
- **Métrica de negocio**: costo del pedido derivado de cada modelo, evaluado con la demanda real:
  `Σ cu·faltante + co·sobrante` con los costos de la configuración; se reporta también contra una
  **política ingenua** (pedir lo vendido la semana anterior ≈ práctica actual).
- **Criterio de reemplazo** (`model_selection`): un challenger reemplaza al champion sólo si
  1) reduce el costo de negocio del backtest en **≥ 3 %**, 2) su WAPE no empeora más de **5 %** relativo,
  3) la cobertura del intervalo 80 % está en **0.80 ± 0.10** (si no se puede medir, no pasa) y
  4) gana en costo en **al menos la mitad de los folds** (no sólo en el total). El costo es el objetivo; WAPE y
  cobertura son guardas para no aceptar un modelo que "gane" en costo con pronósticos sesgados o mal
  calibrados. La decisión, con el backtest de cada candidato, queda en `models/registry.json`.
- **Reentrenamiento cada 4 semanas** (o si no hay modelo vigente). Cada reentrenamiento vuelve a
  correr el backtest y el criterio, y reajusta el champion con los datos más recientes.

## 8. Pedido sugerido

- **Newsvendor**: con costo de faltante `cu` y de sobrante `co`, la cantidad óptima es el cuantil
  `cu/(cu+co)` de la demanda. Costos en `business` de `params.yaml`: no perecibles `cu=1, co=0.25`
  (nivel de servicio 80 %), **perecibles** (pan, lácteos, carnes, frutas y verduras, etc.) `co=0.9`
  (≈ 53 %), y `family_overrides` por familia. **Cambiar los costos cambia el pedido sin tocar código**
  (prueba `test_costs_in_config_change_the_order_without_code_changes`).
- **Suma de cuantiles diarios** en la ventana: exacto si la demanda de días consecutivos es
  perfectamente correlacionada (comonotónica); con correlación menor sobreestima ligeramente la
  dispersión → pedido algo conservador para α > 0.5. Alternativa: simular trayectorias o modelar
  directamente la demanda semanal; se descartó por complejidad y porque el backtest de costo ya mide
  el efecto real de esta aproximación.
- Niveles fuera de [0.10, 0.90] se acotan al cuantil estimado más cercano.
- **Medido** (§10): el nivel de servicio realizado es 0.90 en no perecibles (objetivo 0.80) y 0.51 en
  perecibles (objetivo 0.53). El sobre-servicio de no perecibles es el efecto esperado de sumar
  cuantiles; su costo es bajo (`co` = 0.25) y escalar el pedido no mejoró el costo total en la
  evaluación, por eso no se corrige con un factor ad hoc.

## 9. Versionado y reproducibilidad (Git + DVC)

- **Git**: código, configuración, DAGs, pruebas, punteros `.dvc`, `dvc.yaml/dvc.lock` y métricas pequeñas.
  **DVC**: datos de entrada (Kaggle), bandeja/historia (pipeline `dvc.yaml`), lago, features procesados,
  modelos, pronósticos publicados y monitoreo (`dvc add` desde el DAG semanal + `dvc push`).
- Cada publicación guarda `metadata.json` (commit, `code_dirty`, `code_fingerprint` = hash de
  `src/` + `dags/` + `params.yaml`, hash de configuración, modelo + sha256, hashes del contenido) e
  `inputs_manifest.json` (cada archivo del lago usado + sha256). El commit solo no basta si había
  cambios sin commitear; la huella del código sí identifica exactamente qué se ejecutó.
- `python -m favorita reproduce --cutoff T` recalcula y compara hashes. Como el lago es append-only y el
  corte usa la vista as-of, cualquier versión posterior de los datos reproduce un pronóstico antiguo.
- Remoto DVC local por defecto (`../dvc-storage`, montado en el contenedor) para que funcione sin
  credenciales; se cambia por S3/GDrive con `dvc remote add -d -f`.
- Un fallo de `dvc add/push` no revierte una publicación válida: se registra como error en la tarea.

## 10. Resultados observados (datos reales, bandeja 2017-02-20 → 2017-08-17)

Ejecución completa del flujo con la versión final (`python -m favorita simulate`, misma secuencia que
Airflow; ≈ 36 min): 26 cortes publicados, 7 reentrenamientos.

- **Champion/challenger**: el 2017-02-20 LightGBM **no** reemplazó al baseline (no ganaba en la mayoría
  de los folds); el 2017-03-20 sí (costo de backtest 1 174 248 vs 1 700 848, −31 %; WAPE 0.153 vs
  0.164; cobertura 0.75) y se mantuvo en los 5 reentrenamientos siguientes. Muestra el criterio
  funcionando en ambos sentidos.
- **Seguimiento sobre 24 cortes con la ventana ya observada** (`data/monitoring/metrics_by_cutoff.csv`):
  WAPE medio **0.137** (0.108–0.194), sesgo −0.02, cobertura del intervalo 10–90 % = 0.787 (objetivo
  0.80; 0.80–0.84 desde abril). Calibración por cuantil: P(y ≤ q50) = 0.55, P(y ≤ q90) = 0.88 y
  P(y ≤ q10) = 0.23: el q10 es demasiado alto (intervalo inferior estrecho). El monitoreo lo alerta;
  no afecta el pedido, que usa niveles 0.53 y 0.80.
- **Calidad del pedido**: costo total **7.76 M vs 12.32 M** de la política ingenua (pedir lo vendido la
  semana anterior): **−37 %**; el pedido pierde contra la política ingenua en 4 de 24 semanas, por
  márgenes pequeños, y le gana por mucho en las semanas con cambios (feriados, fin de mes, promociones).
  Nivel de servicio realizado P(demanda ≤ pedido): perecibles **0.51** (objetivo 0.53), no perecibles
  **0.90** (objetivo 0.80; la suma de cuantiles diarios es conservadora, ver §8).
- **Mejora por las correcciones de la evaluación** (§11), misma bandeja: WAPE 0.148 → 0.137, costo del
  pedido 8.45 M → 7.76 M (−8 %); los cortes con domingo faltante pasaron de perder a ganarle a la
  política ingenua (2017-05-15: WAPE 0.248 → 0.125).
- **Situaciones inyectadas** (`reports/source_system_anomalies.csv`): 44 entregas parciales, 27 atrasos,
  18 correcciones, 16 archivos con duplicados y 14 archivos inválidos+reenvío. Los **7 archivos inválidos**
  quedaron en cuarentena y ninguno contaminó el lago; las partes 2, correcciones y reenvíos se
  incorporaron fila a fila al llegar; ningún caso esperado hizo fallar el flujo.
- **Reproducibilidad**: `reproduce` de un corte publicado devuelve `"identical": true`; re-ejecutar un
  corte publicado es *no-op*. Los pronósticos de la corrida local (Windows) y los de Airflow (Linux,
  Docker) para los mismos cortes tienen **el mismo hash de contenido** (el hash se calcula sobre una
  forma canónica del DataFrame, no sobre los bytes del parquet).
- **Airflow 3 en Docker** (validado): el DAG del sistema de origen corre solo, los demás DAGs toman el
  calendario de la bandeja y recorren el periodo con `catchup`; la tarea `dvc_snapshot` hace `dvc add`
  + `dvc push` desde el contenedor.

## 11. Evaluación con agentes calificadores

Se pidió a dos agentes independientes que actuaran como docentes calificadores (uno de operación y
robustez, otro de calidad de las recomendaciones) sobre el criterio oficial de evaluación. Sus
hallazgos reproducidos se corrigieron, con pruebas en `tests/test_operational_scenarios.py` y
`tests/test_features_models.py`:

| Hallazgo | Corrección |
|---|---|
| Pronóstico ≈ 0 si falta el domingo (atraso/entrega parcial) | imputación + entrenamiento con faltantes simulados (§3) |
| Backtest con historia completa no detectaba lo anterior | folds con vista as-of del fold (§7) |
| v1 tardía pisaba a v2 | precedencia por versión (§5) |
| Correcciones válidas rechazadas por filas de tiendas nuevas | integridad referencial fuera del umbral (§4) |
| `train.csv` corto ⇒ todo en cuarentena; fecha 2071 movía la operación | historia mínima y rango robusto (§2) |
| Bandeja externa sin manifiesto no funcionaba | historia desde Kaggle + manifiesto en `data/history` (§2) |
| Disparo manual sin fecha esperaba 24 h | falla inmediata y validación de lunes (§6) |
| Commit registrado sin el código ejecutado | `code_dirty` + `code_fingerprint` (§9) |
| Cobertura del intervalo ocultaba q90 subestimado | calibración por cuantil y alertas en el monitoreo |
| Formatos (`;`, mayúsculas, fecha con hora) iban a cuarentena | normalización (§4) |
| Un ×40 válido disparaba el pedido | tope de valores extremos en features (§4) |
| 25-dic pronosticado con ventas normales | cierres de cadena inferidos de la historia (§6) |

Recomendaciones evaluadas y **no** adoptadas: recalibración conformal de cuantiles (mejora la
calibración de q90 pero empeora el costo del pedido +2 %), features adicionales (sin ganancia en el
backtest) y ventana de 364 días (mejora 2 de 3 cortes, empeora 1; no concluyente y duplica el tiempo).

## 12. Limitaciones y trabajo futuro

- Sin inventario disponible ni restricciones de capacidad/mínimos de pedido del CD.
- La suma de cuantiles es una aproximación (ver §8); podría reemplazarse por cuantiles de la demanda
  agregada de la ventana.
- No se modelan eventos excepcionales (p. ej. terremoto de 2016) más allá de lo que aprenda el modelo.
- Las familias nuevas se rechazan hasta que se agreguen al maestro; una fuente `families` las admitiría.
  Las filas rechazadas por maestro quedan en `*.rejected.csv` pero no se reprocesan solas cuando llega
  el alta (habría que re-entregarlas).
- La métrica de negocio suma unidades con `cu = 1` para todas las familias (un kg de PRODUCE pesa igual
  que una unidad de AUTOMOTIVE); con márgenes por familia bastaría `family_overrides` en `params.yaml`.
- El remoto DVC por defecto es local (`../dvc-storage`); para compartir entre máquinas se configura un
  remoto en la nube (§9).
