# RoadRisk Peru

Aplicacion web con elementos inteligentes para la seguridad vial: estima el riesgo de que un accidente de transito en carretera registre fallecidos (aprendizaje supervisado) y descubre patrones de siniestros con aprendizaje no supervisado.

El proyecto usa:

- **SUTRAN 2020-2021** para entrenamiento supervisado (accidentes con y sin fallecidos).
- **ONSV 2021-2025** como fuente de analisis de siniestros fatales y contexto operativo.
- **FastAPI** para exponer la prediccion y el analisis de patrones como producto web/API.
- **scikit-learn**: Random Forest (supervisado) + K-Means (no supervisado).
- **Model registry manual**: versiones en `models/random_forest/vN/` y `models/clustering/vN/`.
- **Monitoreo de deriva** (PSI + chi-cuadrado) con reportes y alertas.
- **GitHub Actions**: CI en cada push/PR y reentrenamiento automatico diario con comparacion campeon vs retador.

## Ejecucion local

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.roadrisk.train
python -m src.roadrisk.train_clustering
uvicorn app.main:app --reload
```

Luego abrir:

```text
http://127.0.0.1:8000
```

## Endpoints

| Endpoint | Descripcion |
|---|---|
| `GET /` | Interfaz web (prediccion + analisis de patrones). |
| `POST /predict` | Prediccion de riesgo fatal (JSON). |
| `POST /predict-form` | Prediccion desde formulario (HTML). |
| `GET /health` | Estado de salud. |
| `GET /api/v1/clusters` | Perfiles de los clusters descubiertos (k, metricas, patrones). |
| `POST /api/v1/clusters/assign` | Asigna un nuevo siniestro al cluster mas cercano. |
| `GET /api/v1/models/registry` | Estado del registro de versiones de modelos. |
| `GET /api/v1/monitoring/report` | Reporte de deriva de datos (PSI + chi-cuadrado). |

## Entrenamiento y mantenimiento

```powershell
# Modelo supervisado (Random Forest + GridSearchCV)
python -m src.roadrisk.train

# Modelo no supervisado (K-Means con seleccion de k optimo)
python -m src.roadrisk.train_clustering

# Reentrenamiento con comparacion campeon vs retador
python -m src.roadrisk.retrain

# Monitoreo de deriva de datos
python -m src.roadrisk.monitoring
```

El entrenamiento genera:

- `models/roadrisk_model.joblib` y `models/metrics.json` (produccion supervisado).
- `models/clustering_model.joblib` y `models/clustering_metrics.json` (produccion clustering).
- Registro versionado en `models/random_forest/vN/` y `models/clustering/vN/` (modelo, metricas, metadatos).
- Indice `models/registry.json` (versiones por familia + version en produccion).
- `data/processed/sutran_model_ready.csv`.
- `reports/figures/` (matriz de confusion, importancia, elbow, PCA de clusters, tasa fatal por cluster).
- `reports/monitoring/` y `reports/retraining/` (reportes generados por el pipeline).

## Regla de promocion de versiones

El pipeline `retrain` entrena un retador y lo compara contra el campeon en produccion:

- Si el retador mejora `test_recall` (desempate por `test_roc_auc`) -> se promueve a produccion.
- Si empeora o empata -> se conserva el campeon; la version queda registrada en el historial.

## Automatizacion

Workflows en `.github/workflows/`:

- `ci.yml`: instala dependencias, entrena supervisado + clustering, verifica el registro y ejecuta pruebas en cada push o pull request.
- `retrain.yml`: reentrena automaticamente todos los dias (05:00 UTC) y permite ejecucion manual. Flujo: carga datos -> limpia -> entrena retador -> compara campeon/retador -> registra version -> reentrena clustering -> ejecuta monitoreo -> publica artefactos.

Para desplegar: Render usando `render.yaml` (build: `pip install -r requirements.txt && python -m src.roadrisk.train && python -m src.roadrisk.train_clustering`).

## Documentacion

- `reports/informe_tecnico.md`: informe tecnico completo (14 secciones).
- `reports/guion_exposicion.md`: guion de exposicion de 12 minutos + version en ingles.
