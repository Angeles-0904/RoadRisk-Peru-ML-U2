import argparse
import json
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    ONSV_PERSONAS_CSV,
    ONSV_SINIESTROS_CSV,
    PROCESSED_DIR,
    SUTRAN_CSV,
)
from . import registry
from .data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_onsv_summary,
    load_sutran_for_model,
)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(model) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f2 = -1.0
    for threshold in np.linspace(0.15, 0.85, 71):
        pred = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        beta2 = 4
        score = ((1 + beta2) * precision * recall / ((beta2 * precision) + recall)) if (precision + recall) else 0
        if score > best_f2:
            best_f2 = score
            best_threshold = float(threshold)
    return best_threshold, best_f2


def plot_confusion_matrix(y_true, y_pred, output_path) -> None:
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Sin fallecidos", "Con fallecidos"],
        cmap="Blues",
        values_format="d",
    )
    disp.ax_.set_title("Matriz de confusion - RoadRisk Peru")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_feature_importance(best_pipeline: Pipeline, output_path) -> None:
    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    if not hasattr(model, "feature_importances_"):
        return
    names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    idx = np.argsort(importances)[-15:]
    plt.figure(figsize=(9, 6))
    plt.barh(np.array(names)[idx], importances[idx], color="#2563eb")
    plt.title("Variables mas influyentes")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train(sutran_csv=SUTRAN_CSV, promote: bool = True) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_sutran_for_model(sutran_csv)
    df.to_csv(PROCESSED_DIR / "sutran_model_ready.csv", index=False, encoding="utf-8")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )

    candidates = [
        (
            "logistic_regression",
            build_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
            {"model__C": [0.3, 1.0, 3.0]},
        ),
        (
            "random_forest",
            build_pipeline(RandomForestClassifier(random_state=42, class_weight="balanced_subsample", n_jobs=-1)),
            {
                "model__n_estimators": [180],
                "model__max_depth": [8, None],
                "model__min_samples_leaf": [3, 8],
            },
        ),
    ]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    best = None
    leaderboard = []
    for name, pipe, grid in candidates:
        search = GridSearchCV(
            pipe,
            grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        leaderboard.append(
            {
                "model": name,
                "best_cv_roc_auc": float(search.best_score_),
                "best_params": search.best_params_,
            }
        )
        if best is None or search.best_score_ > best["score"]:
            best = {"name": name, "score": search.best_score_, "estimator": search.best_estimator_, "params": search.best_params_}

    estimator = best["estimator"]
    probabilities = estimator.predict_proba(X_test)[:, 1]
    threshold, f2 = find_best_threshold(y_test.to_numpy(), probabilities)
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "positive_rate": float(y.mean()),
        "features": FEATURE_COLUMNS,
        "target": "fatal = FALLECIDOS > 0",
        "best_model": best["name"],
        "best_params": best["params"],
        "leaderboard": leaderboard,
        "decision_threshold": threshold,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "test_f2": float(f2),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    if ONSV_SINIESTROS_CSV.exists() and ONSV_PERSONAS_CSV.exists():
        metrics["onsv_context"] = load_onsv_summary(ONSV_SINIESTROS_CSV, ONSV_PERSONAS_CSV)

    artifact = {
        "pipeline": estimator,
        "threshold": threshold,
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
    }

    # Registro versionado: la version se guarda en models/random_forest/vN/
    # y solo se promueve (punteros planos de produccion) si promote=True.
    version = registry.register(
        "random_forest",
        artifact,
        metrics,
        algorithm=best["name"],
        params=best["params"],
        notes="Modelo supervisado de clasificacion binaria de riesgo fatal (RoadRisk Peru).",
    )
    metrics["version"] = version
    if promote:
        registry.set_production("random_forest", version)

    plot_confusion_matrix(y_test, predictions, FIGURES_DIR / "confusion_matrix.png")
    plot_feature_importance(estimator, FIGURES_DIR / "feature_importance.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sutran-csv", default=str(SUTRAN_CSV))
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Registra la version como retador sin promoverla a produccion",
    )
    args = parser.parse_args()
    metrics = train(args.sutran_csv, promote=not args.no_promote)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
