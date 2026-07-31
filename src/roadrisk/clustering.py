"""Modulo de aprendizaje no supervisado: K-Means para descubrir patrones de siniestros.

Responde preguntas como:
- ¿Existen grupos de accidentes con caracteristicas similares?
  -> Si, se descubren k grupos con la seleccion optima de k (elbow + silhouette + DB).
- ¿Que tipos de accidentes concentran mayor comportamiento de riesgo?
  -> Tasa de siniestros fatales por cluster y riesgo relativo vs. la tasa base.
- ¿Que patrones ocultos aparecen en los datos?
  -> Perfiles de cada cluster (modalidad, departamento, hora, mes, nocturnidad).

Variables: mismas 8 caracteristicas del modelo supervisado.
Preprocesamiento: imputacion + escalado (numericas) y one-hot encoding (categoricas).
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import registry
from .config import CLUSTER_MODEL_PATH, FIGURES_DIR, SUTRAN_CSV
from .data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_sutran_for_model,
)

K_RANGE = range(2, 11)
KMEANS_PARAMS = {"n_init": 10, "random_state": 42}
# silhouette/davies_bouldin son O(n^2): se calculan sobre una submuestra
# (inercia y ajuste final siempre usan el dataset completo).
SILHOUETTE_SAMPLE_SIZE = 4000


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_cluster_preprocessor() -> ColumnTransformer:
    """Preprocesador: imputa, escala y codifica las variables para clustering."""
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _make_one_hot_encoder())]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def prepare_cluster_data(sutran_csv=SUTRAN_CSV):
    """Carga el dataset listo para clustering (features + variable objetivo)."""
    df = load_sutran_for_model(sutran_csv)
    X = df[FEATURE_COLUMNS]
    return df, X


def evaluate_k_values(Z: np.ndarray, k_range=K_RANGE) -> list:
    """Evalua k en un rango con inercia (elbow), silhouette y Davies-Bouldin."""
    rng = np.random.default_rng(42)
    if len(Z) > SILHOUETTE_SAMPLE_SIZE:
        sample_idx = rng.choice(len(Z), size=SILHOUETTE_SAMPLE_SIZE, replace=False)
    else:
        sample_idx = np.arange(len(Z))
    Z_sample = Z[sample_idx]
    results = []
    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            n_init=KMEANS_PARAMS["n_init"],
            random_state=KMEANS_PARAMS["random_state"],
        )
        kmeans.fit(Z)
        labels = kmeans.labels_
        results.append(
            {
                "k": int(k),
                "inertia": float(kmeans.inertia_),
                "silhouette": float(silhouette_score(Z_sample, labels[sample_idx])),
                "davies_bouldin": float(davies_bouldin_score(Z_sample, labels[sample_idx])),
                "silhouette_sample_size": int(len(sample_idx)),
            }
        )
    return results


def elbow_point(k_eval: list) -> int | None:
    """Estimacion del punto del codo: el punto mas alejado de la recta (kmin, kmax)."""
    if len(k_eval) < 3:
        return k_eval[-1]["k"] if k_eval else None
    x1, y1 = float(k_eval[0]["k"]), k_eval[0]["inertia"]
    x2, y2 = float(k_eval[-1]["k"]), k_eval[-1]["inertia"]
    best_k, best_d = None, -1.0
    for item in k_eval[1:-1]:
        x, y = float(item["k"]), item["inertia"]
        d = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / np.hypot(x2 - x1, y2 - y1)
        if d > best_d:
            best_d, best_k = d, item["k"]
    return best_k


def select_optimal_k(k_eval: list) -> tuple[int, int | None]:
    """Selecciona k: criterio principal silhouette maximo, desempate Davies-Bouldin minimo.

    Devuelve (k_optimo, k_codo) para comparar ambos criterios en la evaluacion.
    """
    best = max(k_eval, key=lambda i: (i["silhouette"], -i["davies_bouldin"]))
    return best["k"], elbow_point(k_eval)


def build_profiles(df: pd.DataFrame, labels: np.ndarray, k: int) -> list:
    """Construye el perfil descriptivo de cada cluster (patron descubierto)."""
    dfc = df.copy()
    dfc["_cluster"] = labels
    baseline = float(dfc[TARGET_COLUMN].mean()) or 0.0
    profiles = []
    for c in range(k):
        sub = dfc[dfc["_cluster"] == c]
        size = int(len(sub))
        fatal_rate = float(sub[TARGET_COLUMN].mean()) if size else 0.0
        if fatal_rate >= 0.15:
            risk = "ALTO"
        elif fatal_rate >= 0.10:
            risk = "MEDIO"
        else:
            risk = "BAJO"
        profiles.append(
            {
                "cluster_id": int(c),
                "label": f"CL-{c + 1:02d}",
                "size": size,
                "share": round(size / len(dfc), 4) if len(dfc) else 0.0,
                "fatal_count": int(sub[TARGET_COLUMN].sum()),
                "fatal_rate": round(fatal_rate, 4),
                "riesgo_relativo": round(fatal_rate / baseline, 2) if baseline else 0.0,
                "risk_level": risk,
                "top_modalidad": str(sub["modalidad"].mode().iloc[0]) if size else "—",
                "modalidad_top3": {str(k_): int(v) for k_, v in sub["modalidad"].value_counts().head(3).items()},
                "top_departamento": str(sub["departamento"].mode().iloc[0]) if size else "—",
                "hora_promedio": round(float(sub["hora_siniestro"].mean()), 1) if size else 0.0,
                "mes_promedio": round(float(sub["mes"].mean()), 1) if size else 0.0,
                "noche_rate": round(float(sub["es_noche"].mean()), 4) if size else 0.0,
                "km_promedio": round(float(sub["kilometro"].mean()), 2) if size else 0.0,
            }
        )
    return profiles


def plot_k_selection(k_eval: list, k_opt: int, output_path: Path) -> None:
    """Figura con los tres criterios de seleccion de k (elbow, silhouette, DB)."""
    ks = [i["k"] for i in k_eval]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ks, [i["inertia"] for i in k_eval], marker="o", color="#2563eb")
    axes[0].axvline(k_opt, color="#ff5c5c", ls="--")
    axes[0].set_title("Metodo del codo (inercia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inercia")
    axes[1].plot(ks, [i["silhouette"] for i in k_eval], marker="o", color="#00d4a0")
    axes[1].axvline(k_opt, color="#ff5c5c", ls="--")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")
    axes[2].plot(ks, [i["davies_bouldin"] for i in k_eval], marker="o", color="#f5a623")
    axes[2].axvline(k_opt, color="#ff5c5c", ls="--")
    axes[2].set_title("Indice Davies-Bouldin")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Davies-Bouldin")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_pca_clusters(Z: np.ndarray, labels: np.ndarray, k_opt: int, centroids: np.ndarray, output_path: Path) -> None:
    """Proyeccion PCA 2D de los siniestros coloreados por cluster."""
    pca = PCA(n_components=2, random_state=42)
    Z2 = pca.fit_transform(Z)
    c2 = pca.transform(centroids)
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(Z2[:, 0], Z2[:, 1], c=labels, cmap="tab10", s=12, alpha=0.6)
    ax.scatter(c2[:, 0], c2[:, 1], marker="X", s=180, c="black", edgecolors="white", linewidths=1.2, label="Centroides")
    ax.set_title(f"Clusters de siniestros viales (proyeccion PCA, k={k_opt})")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.legend()
    fig.colorbar(scatter, ticks=range(k_opt))
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_fatal_rate_by_cluster(profiles: list, baseline: float, output_path: Path) -> None:
    """Tasa de siniestros fatales por cluster, con linea base de referencia."""
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [p["label"] for p in profiles]
    rates = [p["fatal_rate"] * 100 for p in profiles]
    colors = [
        "#ff5c5c" if p["risk_level"] == "ALTO" else "#f5a623" if p["risk_level"] == "MEDIO" else "#00d4a0"
        for p in profiles
    ]
    ax.bar(labels, rates, color=colors)
    ax.axhline(baseline * 100, color="#5a7299", ls="--", label=f"Tasa base {baseline * 100:.1f}%")
    ax.set_title("Tasa de siniestros fatales por cluster")
    ax.set_ylabel("% de siniestros con fallecidos")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train_clustering(sutran_csv=SUTRAN_CSV, promote: bool = True) -> dict:
    """Entrena el modelo no supervisado (K-Means) y registra la version.

    - Selecciona el k optimo con elbow + silhouette + Davies-Bouldin.
    - Construye perfiles de patrones por cluster.
    - Registra la version en `models/clustering/` y (si promote) la promueve.
    """
    df, X = prepare_cluster_data(sutran_csv)
    preprocessor = build_cluster_preprocessor()
    Z = preprocessor.fit_transform(X)

    k_eval = evaluate_k_values(Z)
    k_opt, k_elbow = select_optimal_k(k_eval)

    kmeans = KMeans(
        n_clusters=k_opt,
        n_init=KMEANS_PARAMS["n_init"],
        random_state=KMEANS_PARAMS["random_state"],
    )
    kmeans.fit(Z)
    labels = kmeans.labels_
    profiles = build_profiles(df, labels, k_opt)

    best_eval = next(item for item in k_eval if item["k"] == k_opt)
    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "algorithm": "kmeans",
        "k": k_opt,
        "elbow_k": k_elbow,
        "n_features": int(Z.shape[1]),
        "silhouette": best_eval["silhouette"],
        "silhouette_sample_size": int(best_eval["silhouette_sample_size"]),
        "davies_bouldin": best_eval["davies_bouldin"],
        "inertia": best_eval["inertia"],
        "k_evaluation": k_eval,
        "baseline_fatal_rate": round(float(df[TARGET_COLUMN].mean()), 4),
    }

    artifact = {
        "model": kmeans,
        "preprocessor": preprocessor,
        "k": k_opt,
        "features": FEATURE_COLUMNS,
        "profiles": profiles,
        "metrics": metrics,
    }

    version = registry.register(
        "clustering",
        artifact,
        metrics,
        algorithm="kmeans",
        params={"n_clusters": k_opt, "n_init": KMEANS_PARAMS["n_init"], "random_state": KMEANS_PARAMS["random_state"]},
        notes="Modelo no supervisado K-Means de descubrimiento de patrones de siniestros viales (RoadRisk Peru).",
    )
    metrics["version"] = version
    if promote:
        registry.set_production("clustering", version)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_k_selection(k_eval, k_opt, FIGURES_DIR / "cluster_k_selection.png")
    plot_pca_clusters(Z, labels, k_opt, kmeans.cluster_centers_, FIGURES_DIR / "cluster_pca.png")
    plot_fatal_rate_by_cluster(profiles, metrics["baseline_fatal_rate"], FIGURES_DIR / "cluster_fatal_rate.png")
    return metrics


def load_clustering_artifact(model_path=CLUSTER_MODEL_PATH) -> dict:
    """Carga el artefacto de clustering promovido a produccion."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe {model_path}. Ejecuta primero: python -m src.roadrisk.train_clustering"
        )
    return joblib.load(model_path)


def assign_cluster(artifact: dict, record: dict) -> dict:
    """Asigna un nuevo siniestro al cluster mas cercano y devuelve su perfil."""
    features = artifact["features"]
    row = {feature: record.get(feature) for feature in features}
    X = pd.DataFrame([row], columns=features)
    Z = artifact["preprocessor"].transform(X)
    label = int(artifact["model"].predict(Z)[0])
    distance = float(np.min(artifact["model"].transform(Z)[0]))
    profile = artifact["profiles"][label]
    return {"cluster_id": label, "distancia_al_centroide": distance, "perfil": profile}
