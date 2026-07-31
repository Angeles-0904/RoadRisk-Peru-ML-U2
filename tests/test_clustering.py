import pytest

from src.roadrisk.clustering import assign_cluster, load_clustering_artifact
from src.roadrisk.config import CLUSTER_MODEL_PATH

RECORD = {
    "departamento": "LIMA",
    "codigo_via": "PE-1S",
    "kilometro": 24.0,
    "modalidad": "DESPISTE",
    "hora_siniestro": 19,
    "mes": 5,
    "dia_semana": "MONDAY",
    "es_noche": 1,
}


def test_clustering_artifact_metrics():
    if not CLUSTER_MODEL_PATH.exists():
        pytest.skip("Modelo de clustering no entrenado (ejecuta python -m src.roadrisk.train_clustering)")
    artifact = load_clustering_artifact()
    assert artifact["k"] >= 2
    assert artifact["metrics"]["silhouette"] > 0
    assert artifact["metrics"]["davies_bouldin"] > 0
    assert len(artifact["profiles"]) == artifact["k"]


def test_clustering_profiles_fields():
    if not CLUSTER_MODEL_PATH.exists():
        pytest.skip("Modelo de clustering no entrenado")
    artifact = load_clustering_artifact()
    profile = artifact["profiles"][0]
    for field in ("cluster_id", "label", "size", "share", "fatal_rate", "risk_level", "top_modalidad"):
        assert field in profile
    assert profile["risk_level"] in ("ALTO", "MEDIO", "BAJO")


def test_assign_cluster():
    if not CLUSTER_MODEL_PATH.exists():
        pytest.skip("Modelo de clustering no entrenado")
    artifact = load_clustering_artifact()
    result = assign_cluster(artifact, RECORD)
    assert 0 <= result["cluster_id"] < artifact["k"]
    assert result["distancia_al_centroide"] >= 0
    assert result["perfil"]["label"].startswith("CL-")
