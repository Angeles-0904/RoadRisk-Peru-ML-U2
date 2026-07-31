import joblib
import pandas as pd

from .config import MODEL_PATH


def load_model(model_path=MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe {model_path}. Ejecuta primero: python -m src.roadrisk.train"
        )
    return joblib.load(model_path)


def predict_risk(record: dict, model_path=MODEL_PATH) -> dict:
    artifact = load_model(model_path)
    features = artifact["features"]
    row = {feature: record.get(feature) for feature in features}
    X = pd.DataFrame([row], columns=features)
    probability = float(artifact["pipeline"].predict_proba(X)[0, 1])
    threshold = float(artifact["threshold"])
    return {
        "probabilidad_fatal": probability,
        "umbral_operativo": threshold,
        "clasificacion": "ALTO RIESGO" if probability >= threshold else "RIESGO MODERADO",
        "modelo": artifact["metrics"]["best_model"],
    }

