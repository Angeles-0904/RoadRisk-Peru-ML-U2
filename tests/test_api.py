import pytest

from app.main import AccidentInput, cluster_assign, clusters, health, models_registry
from src.roadrisk.config import CLUSTER_MODEL_PATH

PAYLOAD = {
    "departamento": "LIMA",
    "codigo_via": "PE-1S",
    "kilometro": 24.0,
    "modalidad": "DESPISTE",
    "hora_siniestro": 19,
    "mes": 5,
    "dia_semana": "MONDAY",
    "es_noche": 1,
}


def test_health_endpoint():
    assert health()["status"] == "ok"


def test_clusters_endpoint():
    if not CLUSTER_MODEL_PATH.exists():
        pytest.skip("Modelo de clustering no entrenado")
    data = clusters()
    assert data["k"] >= 2
    assert len(data["clusters"]) == data["k"]


def test_cluster_assign_endpoint():
    if not CLUSTER_MODEL_PATH.exists():
        pytest.skip("Modelo de clustering no entrenado")
    result = cluster_assign(AccidentInput(**PAYLOAD))
    assert result["cluster_id"] >= 0
    assert "perfil" in result


def test_registry_endpoint():
    data = models_registry()
    assert "random_forest" in data
    assert "clustering" in data
