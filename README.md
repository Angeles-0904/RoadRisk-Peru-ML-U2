# RoadRisk Peru

Aplicacion web con modelo supervisado para estimar el riesgo de que un accidente de transito en carretera registre fallecidos.

El proyecto usa:

- SUTRAN 2020-2021 para entrenamiento supervisado, porque contiene accidentes con y sin fallecidos.
- ONSV 2021-2025 como fuente de analisis de siniestros fatales y contexto operativo.
- FastAPI para exponer la prediccion como producto web/API.
- scikit-learn para el pipeline de limpieza, entrenamiento y evaluacion.
- GitHub Actions para pruebas, integracion continua y reentrenamiento automatico programado.

## Ejecucion local

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.roadrisk.train
uvicorn app.main:app --reload
```

Luego abrir:

```text
http://127.0.0.1:8000
```

## Entrenamiento

```powershell
python -m src.roadrisk.train --sutran-csv data/raw/sutran_accidentes_2020_2021.csv
```

El entrenamiento genera:

- `models/roadrisk_model.joblib`
- `models/metrics.json`
- `data/processed/sutran_model_ready.csv`
- `reports/figures/confusion_matrix.png`
- `reports/figures/feature_importance.png`

## Automatizacion

Los workflows estan en `.github/workflows/`:

- `ci.yml`: instala dependencias, entrena el modelo y ejecuta pruebas en cada push o pull request.
- `retrain.yml`: reentrena automaticamente una vez al mes y tambien permite ejecucion manual desde GitHub Actions.

Para desplegar, una opcion simple es Render usando `render.yaml`.

