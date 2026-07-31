from src.roadrisk import registry


def _metrics():
    return {"test_recall": 0.7, "test_roc_auc": 0.66, "trained_at_utc": "2026-01-01T00:00:00+00:00"}


def test_register_creates_version_files(tmp_path):
    version = registry.register(
        "random_forest", {"dummy": True}, _metrics(), algorithm="random_forest", params={},
        base_dir=tmp_path, index_path=tmp_path / "registry.json",
    )
    assert version == "v1"
    vdir = tmp_path / "random_forest" / "v1"
    assert (vdir / "model.joblib").exists()
    assert (vdir / "metrics.json").exists()
    assert (vdir / "metadata.json").exists()
    assert (tmp_path / "registry.json").exists()


def test_next_version_increments(tmp_path):
    registry.register("clustering", {"dummy": True}, _metrics(), algorithm="kmeans", params={},
                      base_dir=tmp_path, index_path=tmp_path / "registry.json")
    assert registry.next_version("clustering", base_dir=tmp_path, index_path=tmp_path / "registry.json") == "v2"


def test_first_version_is_auto_production(tmp_path):
    registry.register("random_forest", {"dummy": True}, _metrics(), algorithm="random_forest", params={},
                      base_dir=tmp_path, index_path=tmp_path / "registry.json")
    prod = registry.get_production("random_forest", base_dir=tmp_path, index_path=tmp_path / "registry.json")
    assert prod is not None
    assert prod["version"] == "v1"


def test_set_production_updates_pointer(tmp_path):
    registry.register("random_forest", {"dummy": True}, _metrics(), algorithm="random_forest", params={},
                      base_dir=tmp_path, index_path=tmp_path / "registry.json")
    registry.register("random_forest", {"dummy": True}, _metrics(), algorithm="random_forest", params={},
                      base_dir=tmp_path, index_path=tmp_path / "registry.json")
    registry.set_production("random_forest", "v2", base_dir=tmp_path, index_path=tmp_path / "registry.json", write_flat=False)
    prod = registry.get_production("random_forest", base_dir=tmp_path, index_path=tmp_path / "registry.json")
    assert prod["version"] == "v2"


def test_registry_state_contains_both_families(tmp_path):
    registry.register("random_forest", {"dummy": True}, _metrics(), algorithm="random_forest", params={},
                      base_dir=tmp_path, index_path=tmp_path / "registry.json")
    registry.register("clustering", {"dummy": True}, _metrics(), algorithm="kmeans", params={},
                      base_dir=tmp_path, index_path=tmp_path / "registry.json")
    state = registry.get_registry_state(base_dir=tmp_path, index_path=tmp_path / "registry.json")
    assert "random_forest" in state and "clustering" in state
    assert state["random_forest"]["versions"][0]["version"] == "v1"
