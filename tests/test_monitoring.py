import numpy as np
import pandas as pd

from src.roadrisk.monitoring import build_drift_report, categorical_drift_single, psi


def test_psi_identical_is_zero():
    rng = np.random.default_rng(42)
    x = rng.normal(size=2000)
    assert abs(psi(x, x)) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 2000)
    y = rng.normal(3, 1, 2000)
    assert psi(x, y) > 0.25


def test_categorical_drift_identical():
    rng = np.random.default_rng(7)
    s = rng.choice(["A", "B", "C"], size=1000)
    result = categorical_drift_single(pd.Series(s), pd.Series(s))
    assert result["p_value"] > 0.05
    assert result["max_share_delta"] < 0.05


def _make_frame(rng):
    return pd.DataFrame(
        {
            "kilometro": rng.normal(30, 20, 500),
            "hora_siniestro": rng.integers(0, 24, 500),
            "mes": rng.integers(1, 13, 500),
            "es_noche": rng.integers(0, 2, 500),
            "departamento": rng.choice(["LIMA", "AREQUIPA", "CUSCO"], size=500),
            "codigo_via": rng.choice(["PE-1S", "PE-1N", "PE-3N"], size=500),
            "modalidad": rng.choice(["CHOQUE", "DESPISTE", "ATROPELLO"], size=500),
            "dia_semana": rng.choice(["MONDAY", "SUNDAY"], size=500),
            "fatal": rng.binomial(1, 0.11, 500),
        }
    )


def test_report_structure_identical_data():
    rng = np.random.default_rng(3)
    frame = _make_frame(rng)
    report = build_drift_report(reference_df=frame, current_df=frame.copy(), save=False)
    assert report["status"] == "OK"
    assert {"numeric_drift", "categorical_drift", "target_drift", "alerts", "status"} <= set(report)
    assert "kilometro" in report["numeric_drift"]
    assert "modalidad" in report["categorical_drift"]
    assert report["alerts"] == []
