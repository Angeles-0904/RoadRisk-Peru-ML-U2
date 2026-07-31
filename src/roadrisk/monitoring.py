"""Monitoreo del modelo: deteccion de deriva de datos (data drift) y alertas.

Estrategia basica (sin infraestructura externa):

- Variables numericas: Population Stability Index (PSI) entre la distribucion de
  referencia (entrenamiento) y los datos actuales.
- Variables categoricas: prueba chi-cuadrado de independencia + delta de proporciones.
- Variable objetivo: comparacion de tasa de siniestros fatales (target drift).

Reglas de alerta:
- PSI >= 0.25 -> deriva ALTA.  PSI >= 0.10 -> deriva MODERADA.
- p < 0.01    -> deriva ALTA.  p < 0.05    -> deriva MODERADA.
- |delta tasa de positivos| > 0.02 -> cambio relevante en la variable objetivo.

Uso:
    python -m src.roadrisk.monitor [--reference RUTA] [--current RUTA]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import MONITORING_DIR, REFERENCE_CSV, SUTRAN_CSV
from .data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_sutran_for_model,
)

PSI_ALTO = 0.25
PSI_MODERADO = 0.10
P_ALTO = 0.01
P_MODERADO = 0.05
POSITIVE_DELTA_ALTO = 0.02


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index entre dos distribuciones (0 = identicas)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 2 or len(actual) < 2 or np.nanstd(expected) == 0:
        return 0.0
    breaks = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breaks[0] = -np.inf
    breaks[-1] = np.inf
    e = np.histogram(expected, bins=breaks)[0] / len(expected)
    a = np.histogram(actual, bins=breaks)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def categorical_drift_single(reference, current, top_k: int = 15) -> dict:
    """Detecta cambio en la distribucion de una variable categorica (chi-cuadrado)."""
    ref_series = reference.astype("string").fillna("DESCONOCIDO")
    cur_series = current.astype("string").fillna("DESCONOCIDO")
    ref_counts = ref_series.value_counts(normalize=True)
    cur_counts = cur_series.value_counts(normalize=True)
    top = list(dict.fromkeys(list(ref_counts.head(top_k).index) + list(cur_counts.head(top_k).index)))
    if not top:
        return {"chi2_stat": None, "p_value": 1.0, "max_share_delta": 0.0}

    n_ref, n_cur = float(len(reference)), float(len(current))
    ref_share = ref_counts.reindex(top).fillna(0.0).to_numpy()
    cur_share = cur_counts.reindex(top).fillna(0.0).to_numpy()
    ref_arr = np.append(ref_share * n_ref, max(n_ref - float(ref_share.sum() * n_ref), 0.0))
    cur_arr = np.append(cur_share * n_cur, max(n_cur - float(cur_share.sum() * n_cur), 0.0))
    table = np.vstack([ref_arr, cur_arr]).astype(float)

    try:
        chi2_stat, p_value, _, _ = stats.chi2_contingency(table)
        chi2_stat, p_value = float(chi2_stat), float(p_value)
    except ValueError:
        chi2_stat, p_value = None, 1.0

    return {
        "chi2_stat": chi2_stat,
        "p_value": p_value,
        "max_share_delta": float(np.max(np.abs(ref_share - cur_share))),
    }


def _psi_alert(value: float) -> str:
    if value >= PSI_ALTO:
        return "ALTO"
    if value >= PSI_MODERADO:
        return "MODERADO"
    return "NO"


def _p_alert(p_value: float) -> str:
    if p_value is None:
        return "NO"
    if p_value < P_ALTO:
        return "ALTO"
    if p_value < P_MODERADO:
        return "MODERADO"
    return "NO"


def _load_comparable_frame(path: str | Path) -> pd.DataFrame | None:
    """Carga un dataset para monitoreo, aceptando formato crudo SUTRAN o procesado."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return load_sutran_for_model(path)
    except ValueError:
        # Dataset ya procesado (formato modelo): solo requiere features + objetivo.
        frame = pd.read_csv(path, encoding="utf-8")
        needed = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
        missing = set(needed).difference(frame.columns)
        if missing:
            raise ValueError(f"Dataset no compatible con monitoreo, faltan: {sorted(missing)}")
        return frame[needed]


def build_drift_report(
    reference_df: pd.DataFrame | None = None,
    current_df: pd.DataFrame | None = None,
    reference_path: str | Path | None = None,
    current_path: str | Path | None = None,
    save: bool = True,
    out_dir: str | Path | None = None,
) -> dict:
    """Genera el reporte de deriva comparando la referencia vs. los datos actuales."""
    if reference_df is None:
        path = Path(reference_path) if reference_path else REFERENCE_CSV
        reference_df = _load_comparable_frame(path)
        if reference_df is None:
            reference_df = load_sutran_for_model(SUTRAN_CSV)
    if current_df is None:
        path = Path(current_path) if current_path else SUTRAN_CSV
        current_df = _load_comparable_frame(path)
        if current_df is None:
            current_df = reference_df

    numeric_drift = {}
    for feature in NUMERIC_FEATURES:
        value = psi(reference_df[feature], current_df[feature])
        numeric_drift[feature] = {"psi": value, "alert": _psi_alert(value)}

    categorical_drift = {}
    for feature in CATEGORICAL_FEATURES:
        item = categorical_drift_single(reference_df[feature], current_df[feature])
        item["alert"] = _p_alert(item["p_value"])
        categorical_drift[feature] = item

    ref_pos = float(reference_df[TARGET_COLUMN].mean())
    cur_pos = float(current_df[TARGET_COLUMN].mean())
    target_drift = {
        "reference_positive_rate": round(ref_pos, 4),
        "current_positive_rate": round(cur_pos, 4),
        "delta": round(cur_pos - ref_pos, 4),
    }

    alerts = []
    for feature, item in numeric_drift.items():
        if item["alert"] == "ALTO":
            alerts.append(f"Deriva ALTA en variable numerica '{feature}' (PSI={item['psi']:.3f})")
        elif item["alert"] == "MODERADO":
            alerts.append(f"Deriva MODERADA en variable numerica '{feature}' (PSI={item['psi']:.3f})")
    for feature, item in categorical_drift.items():
        if item["alert"] == "ALTO":
            alerts.append(f"Deriva ALTA en variable categorica '{feature}' (p={item['p_value']:.2e})")
        elif item["alert"] == "MODERADO":
            alerts.append(f"Deriva MODERADA en variable categorica '{feature}' (p={item['p_value']:.2e})")
    if abs(target_drift["delta"]) > POSITIVE_DELTA_ALTO:
        alerts.append(
            f"Cambio relevante en la tasa de siniestros fatales ({target_drift['delta']:+.3f})"
        )

    has_alto = any(i["alert"] == "ALTO" for i in numeric_drift.values()) or any(
        i["alert"] == "ALTO" for i in categorical_drift.values()
    )
    has_moderado = any(i["alert"] in ("ALTO", "MODERADO") for i in numeric_drift.values()) or any(
        i["alert"] in ("ALTO", "MODERADO") for i in categorical_drift.values()
    )
    if has_alto or abs(target_drift["delta"]) > POSITIVE_DELTA_ALTO:
        status = "ACCION_REQUERIDA"
    elif has_moderado:
        status = "REVISION"
    else:
        status = "OK"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
        "numeric_drift": numeric_drift,
        "categorical_drift": categorical_drift,
        "target_drift": target_drift,
        "alerts": alerts,
        "status": status,
    }

    if save:
        out_dir = Path(out_dir or MONITORING_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        (out_dir / f"report_{stamp}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / f"report_{stamp}.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict) -> str:
    lines = [
        "# Reporte de monitoreo - RoadRisk Peru",
        "",
        f"- Generado: {report['generated_at_utc']}",
        f"- Estado: **{report['status']}**",
        f"- Filas de referencia: {report['reference_rows']}",
        f"- Filas actuales: {report['current_rows']}",
        "",
        "## Deriva en variables numericas (PSI)",
        "",
        "| Variable | PSI | Alerta |",
        "|---|---|---|",
    ]
    for feature, item in report["numeric_drift"].items():
        lines.append(f"| {feature} | {item['psi']:.4f} | {item['alert']} |")
    lines += ["", "## Deriva en variables categoricas (chi-cuadrado)", "", "| Variable | p-value | Delta max | Alerta |", "|---|---|---|---|"]
    for feature, item in report["categorical_drift"].items():
        lines.append(f"| {feature} | {item['p_value']:.2e} | {item['max_share_delta']:.4f} | {item['alert']} |")
    lines += ["", "## Deriva de la variable objetivo", "", f"- Tasa de positivos (referencia): {report['target_drift']['reference_positive_rate']}", f"- Tasa de positivos (actual): {report['target_drift']['current_positive_rate']}", f"- Delta: {report['target_drift']['delta']:+.4f}"]
    lines += ["", "## Alertas", ""]
    if report["alerts"]:
        for alert in report["alerts"]:
            lines.append(f"- {alert}")
    else:
        lines.append("- Sin alertas.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=str(REFERENCE_CSV), help="Dataset de referencia (entrenamiento)")
    parser.add_argument("--current", default=str(SUTRAN_CSV), help="Dataset actual (raw)")
    args = parser.parse_args()
    report = build_drift_report(reference_path=args.reference, current_path=args.current)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
