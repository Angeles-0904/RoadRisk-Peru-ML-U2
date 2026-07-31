"""Registro y versionamiento de modelos (estrategia manual).

Estructura de versiones:

    models/
      random_forest/
        v1/
          model.joblib      <- artefacto del modelo supervisado
          metrics.json      <- metricas de evaluacion
          metadata.json     <- fecha, algoritmo, parametros, notas
      clustering/
        v1/
          model.joblib
          metrics.json
          metadata.json
      registry.json         <- indice: versiones por familia + version en produccion

Los archivos planos `models/roadrisk_model.joblib` y `models/clustering_model.joblib`
se mantienen como punteros a la version promovida a produccion, preservando la
compatibilidad con la aplicacion existente (no se elimina funcionalidad).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .config import (
    CLUSTER_METRICS_PATH,
    CLUSTER_MODEL_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_REGISTRY_ROOT,
    REGISTRY_INDEX_PATH,
)


def _resolve_paths(base_dir=None, index_path=None):
    """Resuelve directorio base e indice, usando los valores de config por defecto."""
    if base_dir is None:
        base_dir = MODEL_REGISTRY_ROOT
        if index_path is None:
            index_path = REGISTRY_INDEX_PATH
    else:
        base_dir = Path(base_dir)
        if index_path is None:
            index_path = base_dir / "registry.json"
    return base_dir, Path(index_path)

FAMILIES = ("random_forest", "clustering")

# Punteros planos de produccion por familia de modelo
FLAT_POINTERS = {
    "random_forest": (MODEL_PATH, METRICS_PATH),
    "clustering": (CLUSTER_MODEL_PATH, CLUSTER_METRICS_PATH),
}


def _load_index(index_path: Path) -> dict:
    if index_path.exists():
        try:
            return json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_index(index: dict, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def next_version(family: str, base_dir=None, index_path=None) -> str:
    """Devuelve la siguiente version disponible (v1, v2, ...) para una familia."""
    base_dir, index_path = _resolve_paths(base_dir, index_path)
    index = _load_index(index_path)
    versions = index.get(family, {}).get("versions", [])
    numbers = [int(v.removeprefix("v")) for v in versions if str(v).startswith("v")] or [0]
    return f"v{max(numbers) + 1}"


def register(family, artifact, metrics, algorithm, params, notes="", base_dir=None, index_path=None) -> str:
    """Registra una nueva version del modelo y devuelve su identificador.

    Escribe model.joblib, metrics.json y metadata.json en el directorio de la
    version y actualiza el indice `registry.json`. Si es la primera version de
    la familia, se marca automaticamente como produccion.
    """
    base_dir, index_path = _resolve_paths(base_dir, index_path)
    version = next_version(family, base_dir=base_dir, index_path=index_path)

    vdir = base_dir / family / version
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, vdir / "model.joblib")

    metrics = dict(metrics)
    metrics["version"] = version
    (vdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metadata = {
        "version": version,
        "family": family,
        "algorithm": algorithm,
        "params": params,
        "trained_at_utc": metrics.get("trained_at_utc"),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }
    (vdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    index = _load_index(index_path)
    index.setdefault(family, {}).setdefault("versions", [])
    if version not in index[family]["versions"]:
        index[family]["versions"].append(version)
    index[family]["latest"] = version
    if index[family].get("production") is None:
        index[family]["production"] = version
    _save_index(index, index_path)
    return version


def set_production(family, version, base_dir=None, index_path=None, write_flat=True) -> None:
    """Promueve una version como produccion y actualiza los punteros planos."""
    base_dir, index_path = _resolve_paths(base_dir, index_path)
    index = _load_index(index_path)
    index.setdefault(family, {})
    index[family]["production"] = version
    _save_index(index, index_path)

    if write_flat and family in FLAT_POINTERS:
        vdir = base_dir / family / version
        model_file = vdir / "model.joblib"
        metrics_file = vdir / "metrics.json"
        if model_file.exists():
            joblib.dump(joblib.load(model_file), FLAT_POINTERS[family][0])
        if metrics_file.exists():
            FLAT_POINTERS[family][1].write_text(
                metrics_file.read_text(encoding="utf-8"), encoding="utf-8"
            )


def get_version_info(family, version, base_dir=None) -> dict | None:
    """Devuelve la informacion completa de una version (rutas, metricas, metadatos)."""
    base_dir, _ = _resolve_paths(base_dir, None)
    vdir = base_dir / family / version
    if not ((vdir / "model.joblib").exists() and (vdir / "metrics.json").exists()):
        return None
    info = {
        "version": version,
        "family": family,
        "model_path": vdir / "model.joblib",
        "metrics_path": vdir / "metrics.json",
        "metadata_path": vdir / "metadata.json",
        "metrics": json.loads((vdir / "metrics.json").read_text(encoding="utf-8")),
    }
    if (vdir / "metadata.json").exists():
        info["metadata"] = json.loads((vdir / "metadata.json").read_text(encoding="utf-8"))
    else:
        info["metadata"] = {"version": version, "family": family}
    return info


def get_production(family, base_dir=None, index_path=None) -> dict | None:
    """Devuelve la informacion de la version en produccion de una familia (o None)."""
    base_dir, index_path = _resolve_paths(base_dir, index_path)
    version = _load_index(index_path).get(family, {}).get("production")
    if not version:
        return None
    return get_version_info(family, version, base_dir=base_dir)


def list_versions(family, base_dir=None, index_path=None) -> list:
    base_dir, index_path = _resolve_paths(base_dir, index_path)
    return _load_index(index_path).get(family, {}).get("versions", [])


def get_registry_state(base_dir=None, index_path=None) -> dict:
    """Estado completo del registro para exponerlo via API."""
    base_dir, index_path = _resolve_paths(base_dir, index_path)
    index = _load_index(index_path)
    state = {}
    for family in FAMILIES:
        versions = []
        for version in index.get(family, {}).get("versions", []):
            info = get_version_info(family, version, base_dir=base_dir)
            if info:
                versions.append(
                    {
                        "version": version,
                        "metadata": info["metadata"],
                        "metrics": info["metrics"],
                    }
                )
        state[family] = {
            "production": index.get(family, {}).get("production"),
            "latest": index.get(family, {}).get("latest"),
            "versions": versions,
        }
    return state
