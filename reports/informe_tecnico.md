# Informe tecnico - RoadRisk Peru (Unidad 2)

## 0. Correspondencia con la rubrica

Este informe cubre explícitamente los criterios de evaluacion de la Unidad 2:

| Criterio de la rubrica | Pts | Seccion de este informe |
|---|---|---|
| Herramientas, plataformas y aplicaciones para despliegue, mantenimiento e IC | 1 | Seccion 15 |
| Organizacion del codigo fuente | 1 | Seccion 2 |
| Consideraciones de despliegue inicial | 1 | Seccion 9 |
| Flujos de mantenimiento e integracion continua | 2 | Secciones 10 y 11 |
| Video grabado de la exposicion (URL) | 3 | Seccion 16 |

## 1. Descripcion general del sistema actualizado

RoadRisk Peru es una aplicacion web/API que estima la probabilidad de que un accidente de transito en carretera registre fallecidos y que, ademas, descubre patrones ocultos de siniestros mediante aprendizaje no supervisado.

La version 2.0.0 incorpora:

- **Modelo supervisado** (Random Forest): clasificacion binaria de riesgo fatal.
- **Modelo no supervisado** (K-Means): descubrimiento de grupos de accidentes con caracteristicas similares y su nivel de riesgo asociado.
- **Registro y versionamiento de modelos**: historial de versiones por familia de modelo con fecha, algoritmo, parametros y metricas.
- **Monitoreo de deriva de datos**: deteccion de cambios de distribucion (PSI + chi-cuadrado) con alertas.
- **Retraining campeon vs retador**: pipeline que entrena una nueva version, la compara con la de produccion y promueve solo si mejora.
- **CI/CD automatizado**: integracion continua y reentrenamiento diario programado.

El objetivo no es reemplazar la evaluacion policial o medica, sino priorizar alertas y apoyar decisiones tempranas de respuesta.

**URL de la aplicacion en produccion**: `https://roadrisk-peru-ml-u2.onrender.com`

## 2. Arquitectura del sistema y organizacion del codigo fuente

```text
+---------------------+        +--------------------------+
|  Cliente (navegador) |        |   GitHub Actions (CI/CD)  |
|  - /predict-form     |        |  - ci.yml (push/PR)      |
|  - Analisis patrones |        |  - retrain.yml (diario)  |
+----------+----------+        +------------+-------------+
           |                               |
           v                               v
+----------+-------------------------------v----------------+
|                    FastAPI (app/main.py)                  |
|  GET /                 UI (tabs: riesgo + patrones)      |
|  POST /predict         Prediccion supervisada            |
|  POST /predict-form    Prediccion desde formulario       |
|  GET /api/v1/clusters          Perfiles de clusters      |
|  POST /api/v1/clusters/assign  Asignar siniestro a cluster|
|  GET /api/v1/models/registry   Estado del registro       |
|  GET /api/v1/monitoring/report Reporte de deriva         |
+----+----------------+----------------+-------------------+
     |                |                |
     v                v                v
+-----------+  +-------------+  +----------------+
| supervised |  | clustering  |  |   monitoring   |
| src/roadrisk|  | src/roadrisk|  | src/roadrisk   |
| predict.py |  | clustering  |  | monitoring.py   |
| train.py   |  | .py         |  | registry.py     |
+-----+-----+  +------+------+  +--------+--------+
      |               |                  |
      v               v                  v
+-----+---------------+------------------+--------+
|            models/ (registro versionado)         |
|  random_forest/v1..vN/  clustering/v1..vN/      |
|  roadrisk_model.joblib  clustering_model.joblib |
|  metrics.json           clustering_metrics.json |
|  registry.json                                  |
+-------------------------------------------------+
```

Capas:

1. **Presentacion**: interfaz web con dos pestanas (Prediccion de riesgo y Analisis de patrones).
2. **API**: FastAPI con endpoints de prediccion, clustering, registro y monitoreo.
3. **Logica**: paquete `src/roadrisk` con modulos separados por responsabilidad.
4. **Datos**: `data/raw` (fuentes originales) y `data/processed` (datos limpios).
5. **Modelos**: registro versionado en `models/` con punteros planos de produccion para compatibilidad.

Organizacion del codigo fuente (separacion de responsabilidades):

```text
app/
  main.py                    Aplicacion FastAPI (UI + API)
src/roadrisk/
  config.py                  Rutas y constantes globales
  data.py                    Carga, limpieza e ingenieria de caracteristicas
  train.py                   Entrenamiento supervisado (GridSearchCV) + registro
  predict.py                 Carga del modelo y prediccion
  clustering.py              Modelo no supervisado K-Means, perfiles y asignacion
  train_clustering.py        CLI de entrenamiento de clustering
  registry.py                Registro y versionamiento de modelos
  retrain.py                 Retraining con comparacion campeon vs retador
  monitoring.py              Deteccion de deriva de datos y reportes
data/
  raw/                       CSV originales (SUTRAN, ONSV)
  processed/                 Datos limpios generados
models/
  random_forest/vN/          Versiones del modelo supervisado
  clustering/vN/             Versiones del modelo no supervisado
  registry.json              Indice de versiones y produccion
reports/
  figures/                   Figuras (confusion, importancia, clusters)
  monitoring/                Reportes de deriva
  retraining/                Reportes de comparacion campeon/retador
tests/                       Pruebas automatizadas (17)
.github/workflows/           CI y reentrenamiento automatico
Dockerfile, render.yaml      Despliegue
```

| Modulo | Responsabilidad |
|---|---|
| `config.py` | Rutas y constantes globales. |
| `data.py` | Carga, limpieza e ingenieria de caracteristicas. |
| `train.py` | Entrenamiento supervisado (GridSearchCV) + registro. |
| `predict.py` | Carga del modelo y prediccion. |
| `clustering.py` | Modelo no supervisado K-Means, perfiles y asignacion. |
| `train_clustering.py` | CLI de entrenamiento de clustering. |
| `registry.py` | Registro y versionamiento de modelos. |
| `retrain.py` | Retraining con comparacion campeon vs retador. |
| `monitoring.py` | Deteccion de deriva de datos y reportes. |

## 3. Dataset

Fuentes:

- `data/raw/sutran_accidentes_2020_2021.csv`: base SUTRAN de accidentes en carreteras con fallecidos y heridos. Se usa para entrenamiento (supervisado y no supervisado) porque contiene casos con y sin fallecidos.
- `data/raw/onsv_siniestros_fatales_2021_2025.csv`: base ONSV de siniestros fatales (contexto).
- `data/raw/onsv_personas_involucradas_2021_2025.csv`: base ONSV de personas involucradas (contexto).

Filas modeladas: **8,067**. Tasa de siniestros con fallecidos: **11.71%** (clase minoritaria).

## 4. Ingenieria de caracteristicas

Las 8 variables de entrada (identicas para ambos modelos):

- Categoricas: `departamento`, `codigo_via`, `modalidad`, `dia_semana`.
- Numericas: `kilometro`, `hora_siniestro`, `mes`, `es_noche`.

Variable objetivo (solo supervisado):

```text
fatal = 1 si FALLECIDOS > 0
fatal = 0 si FALLECIDOS = 0
```

Preprocesamiento (pipeline):

- Lectura `latin1` con separador `;` y normalizacion de nombres de columnas.
- Conversion de fechas `YYYYMMDD` y extraccion de hora, mes y dia de semana.
- Limpieza numerica del kilometraje y del horario.
- Imputacion de valores faltantes.
- Escalado (StandardScaler) de variables numericas.
- One-hot encoding de variables categoricas (categorias raras colapsadas con `min_frequency=5`).

Se evita usar la cantidad de fallecidos como entrada (fuga de informacion). El numero de heridos tambien se excluye para permitir una prediccion temprana.

## 5. Modelo supervisado

- Algoritmo: **Random Forest** con `class_weight="balanced_subsample"`.
- Comparado contra regresion logistica balanceada.
- Hiperparametros seleccionados (GridSearchCV): `n_estimators=180`, `max_depth=8`, `min_samples_leaf=8`.
- Optimizacion: `GridSearchCV` con validacion cruzada estratificada de 3 particiones y metrica `roc_auc`.
- Umbral operativo: **0.42**, optimizado con F2 para priorizar recall (en seguridad vial es mas grave no alertar un caso fatal que generar una alerta preventiva).

## 6. Modelo no supervisado

- Algoritmo: **K-Means** sobre las mismas 8 variables (one-hot + escalado).
- Seleccion de k optimo con tres criterios: **metodo del codo** (inercia), **Silhouette Score** y **indice Davies-Bouldin**. Criterio principal: maximo silhouette; desempate: menor Davies-Bouldin.
- Resultado: **k = 2** (el codo por inercia sugiere k=4, pero silhouette y Davies-Bouldin son maximos/mínimos en k=2, que ademas es el k mas interpretable).

Evaluacion de k (silhouette sobre submuestra de 4,000 filas para mantener el build rapido):

| k | Inercia | Silhouette | Davies-Bouldin |
|---|---|---|---|
| 2 | 46153.0 | **0.2330** | **1.6603** |
| 3 | 41406.3 | 0.1960 | 1.8455 |
| 4 | 37686.5 | 0.1519 | 1.9782 |
| 5 | 35927.2 | 0.1473 | 2.0487 |
| 6 | 34622.5 | 0.1347 | 2.0859 |
| 7 | 33354.5 | 0.1159 | 2.1854 |
| 8 | 32328.6 | 0.1090 | 2.3642 |
| 9 | 31597.1 | 0.1067 | 2.4257 |
| 10 | 30985.9 | 0.1008 | 2.4177 |

Patrones descubiertos (perfiles de cluster):

| Cluster | Casos | % | Tasa fatal | Riesgo relativo | Nivel | Modalidad dominante | Hora media | Nocturno |
|---|---|---|---|---|---|---|---|---|
| CL-01 | 5,920 | 73.4% | 11.33% | 0.97x | MEDIO | DESPISTE | 9.8h | 0% |
| CL-02 | 2,147 | 26.6% | 12.76% | 1.09x | MEDIO | CHOQUE | 20.2h | 100% |

Interpretacion: **CL-02 es el patron nocturno** (100% de siniestros de noche, hora media 20.2h, dominado por CHOQUE) y concentra **9% mas riesgo relativo** que el patron diurno CL-01. Esto responde las preguntas del analisis:

- Existen grupos de accidentes con caracteristicas similares? Si, dos patrones claramente separados por horario/nocturnidad.
- Que tipos de accidentes tienen mayor comportamiento de riesgo? El patron nocturno de choques (CL-02).
- Que patrones ocultos aparecen? La separacion diurno/nocturno, que no era visible en el analisis descriptivo simple.

Figuras generadas: `reports/figures/cluster_k_selection.png`, `cluster_pca.png` (proyeccion PCA), `cluster_fatal_rate.png`.

## 7. Entrenamiento

Proceso reproducido en la validacion (2026-07-31):

1. Carga y limpieza de SUTRAN (`load_sutran_for_model`).
2. Guardado de `data/processed/sutran_model_ready.csv`.
3. Supervisado: GridSearchCV sobre LogisticRegression y RandomForest; seleccion por ROC AUC; umbral por F2.
4. No supervisado: preprocesamiento (imputacion + escala + one-hot, 120 features), evaluacion de k=2..10, ajuste de K-Means con `random_state=42` y `n_init=10`, construccion de perfiles.
5. Registro de versiones en `models/` (ambas familias).
6. Generacion de figuras y metricas JSON.

Reproducibilidad: `random_state=42` fijado en splits, modelos y seleccion de submuestra.

## 8. Evaluacion

Modelo supervisado (version v1, conjunto de prueba 22%):

| Metrica | Valor |
|---|---|
| ROC AUC | **0.6675** |
| Recall | **0.7356** |
| Precision | 0.1581 |
| F1 | 0.2602 |
| Accuracy | 0.5099 |
| Umbral operativo | 0.42 |

La precision es baja porque la clase fatal es minoritaria (11.7%). Para una aplicacion de seguridad vial se priorizo recall: es mas grave no alertar un caso fatal que generar una alerta preventiva adicional.

Modelo no supervisado:

- Silhouette: **0.2330** (estructura de clusters moderada; tipico en datos viales de alta dimensionalidad).
- Davies-Bouldin: **1.6603** (cuanto menor, mejor separacion).
- Inercia k=2: 46153.0.

Monitoreo (referencia = datos de entrenamiento vs datos actuales):

- PSI = 0.0 en todas las variables numericas.
- p = 1.0 en todas las variables categoricas.
- Delta de tasa de positivos: 0.0.
- Estado: **OK**, sin alertas (esperado: los datos actuales son el mismo corte de entrenamiento; el pipeline detecta deriva cuando llega un CSV nuevo).

## 9. Despliegue y consideraciones de despliegue inicial

La aplicacion esta **desplegada en produccion** en Render (plan free):

- **URL**: `https://roadrisk-peru-ml-u2.onrender.com`
- **Repositorio**: `https://github.com/Angeles-0904/RoadRisk-Peru-ML-U2`

Despliegue por contenedor (Dockerfile):

- Imagen base `python:3.11-slim`.
- Build: `pip install -r requirements.txt && python -m src.roadrisk.train && python -m src.roadrisk.train_clustering` (entrena ambos modelos dentro del build; la app queda lista sin pasos manuales).
- Start: `CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]` — Render inyecta el puerto via `$PORT` y `exec` deja a uvicorn como PID 1 (apagado limpio).

Consideraciones de despliegue inicial:

1. Crear un repositorio GitHub con el codigo fuente.
2. En Render: "New Web Service" o "Blueprint" conectado al repositorio.
3. Seleccionar el plan free; no requiere variables de entorno adicionales.
4. En el primer build Render instala dependencias y entrena ambos modelos (unos minutos).
5. Verificar con `GET /health` y los endpoints `/api/v1/*`.
6. En el plan free, tras 15 min sin visitas el servicio se suspende; la primera peticion tras la suspension tarda unos segundos.

Nota operativa: cada build de despliegue ejecuta el entrenamiento con promocion (`promote=True`), por lo que cada deploy puede registrar y promover una version nueva (v2, v3, ...) aunque los datos no hayan cambiado. Esto replica el comportamiento de la Unidad 1 (build que reentrena) y demuestra el versionamiento en accion; si se prefiere evitar la creacion de versiones en cada deploy, basta ejecutar `python -m src.roadrisk.retrain` en el build, que promueve solo si el retador mejora al campeon.

## 10. Integracion continua (CI)

`.github/workflows/ci.yml` (se ejecuta en cada push y pull request):

1. Instala dependencias (`pip install -r requirements.txt`).
2. Entrena el modelo supervisado (`python -m src.roadrisk.train`).
3. Entrena el modelo no supervisado (`python -m src.roadrisk.train_clustering`).
4. Verifica el registro (`registry.json` con version en produccion para ambas familias).
5. Ejecuta la suite de pruebas (17 tests).
6. Publica artefactos (modelos y figuras).

## 11. Mantenimiento automatico (reentrenamiento)

`.github/workflows/retrain.yml` (diario 05:00 UTC + ejecucion manual desde GitHub Actions):

1. Instala dependencias.
2. Ejecuta `python -m src.roadrisk.retrain`:
   - Carga la version en produccion (campeon).
   - Entrena un retador con los datos disponibles (sin promoverlo aun).
   - Compara: si el retador mejora `test_recall` (desempate `test_roc_auc`) se promueve; si empeora o empata, se conserva el campeon.
   - Reentrena el modelo no supervisado (K-Means) internamente.
   - Genera reporte de comparacion en `reports/retraining/`.
3. Ejecuta el monitoreo de deriva (`python -m src.roadrisk.monitoring`) y guarda el reporte en `reports/monitoring/`.
4. Ejecuta pruebas y publica artefactos.

Resultado de la validacion del 2026-07-31:

- Campeon v1: recall 0.7356, ROC AUC 0.6675.
- Retador v2 (mismos datos): recall 0.7356, ROC AUC 0.6675.
- Decision: **KEPT_CHAMPION** (el retador no mejora; se conserva el campeon, v2 queda registrado).

Regla de negocio implementada:

```text
Si recall_retador > recall_campeon   -> promover retador
Si recall_retador <= recall_campeon  -> conservar campeon
```

## 12. Versionamiento de modelos

Registro manual (`src/roadrisk/registry.py`):

```text
models/
  random_forest/
    v1/  model.joblib  metrics.json  metadata.json
    v2/  model.joblib  metrics.json  metadata.json
    v3/  model.joblib  metrics.json  metadata.json
  clustering/
    v1/  model.joblib  metrics.json  metadata.json
    v2/  model.joblib  metrics.json  metadata.json
    v3/  model.joblib  metrics.json  metadata.json
  registry.json        <- versiones por familia + version en produccion
```

`metadata.json` registra: version, familia, algoritmo, parametros, fecha de entrenamiento, fecha de registro y notas.

**Estado actual del registro en produccion** (consultado via `GET /api/v1/models/registry`):

- `random_forest`: produccion **v3** (versiones: v1, v2, v3).
- `clustering`: produccion **v3** (versiones: v1, v2, v3).

Nota de coherencia: las versiones v3 en produccion provienen del build de despliegue en Render, que reentrena con promocion en cada deploy (ver seccion 9). La validacion local de reentrenamiento (seccion 11) demostro la regla campeon/retador conservando v1; ambos hechos son consistentes: el pipeline de mantenimiento solo promueve si el retador mejora, mientras el build de despliegue registra una version nueva en cada deploy.

Los archivos planos `roadrisk_model.joblib` y `clustering_model.joblib` son punteros a la version en produccion, lo que preserva la compatibilidad con la app (no se elimino funcionalidad).

Se eligio la estructura manual (sin MLflow) por: cero infraestructura adicional, compatibilidad total con Render free y despliegue por contenedor, y facilidad de explicacion y auditoria en el curso. El informe tecnico de la Unidad 1 ya exigia "versionar cambios con nombres claros"; ahora el sistema lo hace de forma automatica.

## 13. Pruebas

Suite automatizada (`tests/`, 17 pruebas, todas en verde):

| Archivo | Cobertura |
|---|---|
| `test_api.py` | Health, clusters, asignacion de cluster, registro. |
| `test_data_pipeline.py` | Carga SUTRAN: features + target presentes, variable binaria. |
| `test_clustering.py` | Artefacto de clustering, k>=2, perfiles, asignacion. |
| `test_registry.py` | Registro de versiones, incremento, promocion, estado del registro. |
| `test_monitoring.py` | PSI identico=0, PSI detecta deriva, chi-cuadrado, estructura del reporte. |

Los tests de clustering y API usan `pytest.skip` si el modelo no esta entrenado, para permitir ejecutar la suite antes del primer entrenamiento (en CI siempre se entrena primero).

**Pruebas de funcionamiento del mantenimiento e integracion continua** (criterio de la rubrica):

- El pipeline CI se ejecuto de extremo a extremo (instalacion -> entrenamiento de ambos modelos -> verificacion del registro -> 17 pruebas -> artefactos).
- El pipeline de reentrenamiento se ejecuto localmente y produjo: entrenamiento del retador, comparacion campeon/retador con decision `KEPT_CHAMPION`, registro de la nueva version y reporte en `reports/retraining/`.
- El monitoreo de deriva se ejecuto y genero reporte con estado `OK` en `reports/monitoring/`.
- La aplicacion desplegada responde correctamente en todos los endpoints verificados (`/health`, `/`, `/api/v1/clusters`, `/api/v1/models/registry`, `/api/v1/monitoring/report`, `POST /predict`, `POST /api/v1/clusters/assign`).

## 14. Manual para equipo TI

### Puesta en marcha local

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.roadrisk.train
python -m src.roadrisk.train_clustering
uvicorn app.main:app --reload
```

### Comandos de mantenimiento

| Tarea | Comando |
|---|---|
| Reentrenar supervisado + comparar | `python -m src.roadrisk.retrain` |
| Reentrenar supervisado (promover) | `python -m src.roadrisk.train` |
| Reentrenar clustering | `python -m src.roadrisk.train_clustering` |
| Monitorear deriva | `python -m src.roadrisk.monitoring` |
| Ver estado del registro | `GET /api/v1/models/registry` |
| Ver reporte de deriva | `GET /api/v1/monitoring/report` |

### Incorporar datos nuevos

1. Colocar el nuevo CSV en `data/raw/` (respetando el esquema SUTRAN).
2. Ejecutar `python -m src.roadrisk.retrain` (o esperar el cron diario).
3. Revisar `reports/retraining/comparison_*.json` para ver la decision (promovido/conservado).
4. Revisar `reports/monitoring/report_*.json` para ver alertas de deriva.
5. Si hay deriva ALTA, investigar el cambio de distribucion antes de liberar el modelo.

### Operacion en produccion

- El despliegue (Render/Docker) entrena ambos modelos en el build; no requiere pasos manuales.
- El cron diario reentrena, compara y registra; las versiones historicas siempre quedan disponibles.
- Para un despliegue experimental de una version concreta, promoverla con `registry.set_production` o reentrenar con `--no-promote` y luego promover si las metricas lo justifican.

### Despliegue en Render

1. Subir el proyecto a GitHub.
2. Conectar el repositorio con Render.
3. Render ejecuta el build (dependencias + entrenamiento de ambos modelos) e inicia Uvicorn.
4. Verificar `GET /health` y los endpoints `/api/v1/*`.

## 15. Herramientas, plataformas y aplicaciones necesarias

| Herramienta/Plataforma | Uso en el proyecto | Version |
|---|---|---|
| Python | Lenguaje de programacion de todo el sistema | 3.11 |
| FastAPI | Framework web (UI + API REST) | >=0.110 |
| Uvicorn | Servidor ASGI que levanta la aplicacion | >=0.27 |
| scikit-learn | Modelos (Random Forest, K-Means) y metrica (GridSearchCV, silhouette, Davies-Bouldin) | >=1.3 |
| pandas / numpy | Procesamiento de datos | pandas>=2.0, numpy>=1.24 |
| scipy | Prueba chi-cuadrado del monitoreo | >=1.10 |
| joblib | Serializacion de modelos | >=1.3 |
| matplotlib | Figuras (matriz de confusion, clusters, codo) | >=3.7 |
| pytest | Pruebas automatizadas | >=8.0 |
| Git / GitHub | Control de versiones y repositorio remoto | - |
| GitHub Actions | Integracion continua y reentrenamiento automatico | - |
| Docker | Contenedor de la aplicacion | - |
| Render | Plataforma de despliegue en produccion (plan free) | - |

## 16. Video grabado de la exposicion

URL del video (maximo 12 minutos):

```text
[PENDIENTE - insertar URL del video grabado de la exposicion]
```

Nota: el video debe incluir la explicacion tecnica del producto, el funcionamiento de la aplicacion en ingles, el entrenamiento del modelo, el despliegue, los pipelines de mantenimiento e integracion continua, y las pruebas de funcionamiento. El guion de la exposicion esta en `reports/guion_exposicion.md` (estructura de 12 minutos + version en ingles).
