"""Retraining con comparacion campeon/retador (champion/challenger).

Flujo:
1. Carga la version en produccion (campeon) del registro.
2. Entrena un retador con los datos actuales (sin promoverlo todavia).
3. Compara metricas: campeon vs. retador (primary: recall, tie-break: ROC AUC).
4. Regla de negocio:
   - Si el retador MEJORA  -> se promueve a produccion (nueva version).
   - Si el retador EMPEORA -> se mantiene el campeon (la version queda registrada).
5. Reentrena tambien el modelo no supervisado (K-Means) y lo registra.
6. Genera el reporte de comparacion en `reports/retraining/`.

Uso:
    python -m src.roadrisk.retrain [--sutran-csv RUTA]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import registry
from .clustering import train_clustering
from .config import RETRAINING_DIR, SUTRAN_CSV
from .train import train

PRIMARY_METRIC = "test_recall"
SECONDARY_METRIC = "test_roc_auc"
EPSILON = 1e-6


def compare_and_promote(sutran_csv=SUTRAN_CSV) -> dict:
    champion = registry.get_production("random_forest")

    # 1) Entrenar al retador (registrado pero NO promovido)
    challenger_metrics = train(sutran_csv, promote=False)
    challenger_version = challenger_metrics["version"]

    # 2) Comparar y decidir
    if champion is None:
        registry.set_production("random_forest", challenger_version)
        decision = "PROMOTED_FIRST_VERSION"
        champion_metrics = None
    else:
        champion_metrics = champion["metrics"]
        challenger_better = (
            challenger_metrics[PRIMARY_METRIC] > champion_metrics[PRIMARY_METRIC] + EPSILON
        )
        if not challenger_better and abs(challenger_metrics[PRIMARY_METRIC] - champion_metrics[PRIMARY_METRIC]) <= EPSILON:
            challenger_better = (
                challenger_metrics[SECONDARY_METRIC] > champion_metrics[SECONDARY_METRIC] + EPSILON
            )
        decision = "PROMOTED" if challenger_better else "KEPT_CHAMPION"
        if challenger_better:
            registry.set_production("random_forest", challenger_version)

    # 3) Reentrenar y promover el modelo no supervisado (los patrones cambian con los datos)
    cluster_metrics = train_clustering(sutran_csv, promote=True)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "decision": decision,
        "champion": (
            {
                "version": champion["version"],
                PRIMARY_METRIC: champion_metrics[PRIMARY_METRIC],
                SECONDARY_METRIC: champion_metrics[SECONDARY_METRIC],
            }
            if champion is not None
            else None
        ),
        "challenger": {
            "version": challenger_version,
            PRIMARY_METRIC: challenger_metrics[PRIMARY_METRIC],
            SECONDARY_METRIC: challenger_metrics[SECONDARY_METRIC],
        },
        "clustering": {
            "version": cluster_metrics["version"],
            "k": cluster_metrics["k"],
            "silhouette": cluster_metrics["silhouette"],
            "davies_bouldin": cluster_metrics["davies_bouldin"],
        },
    }

    RETRAINING_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (RETRAINING_DIR / f"comparison_{stamp}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sutran-csv", default=str(SUTRAN_CSV))
    args = parser.parse_args()
    summary = compare_and_promote(args.sutran_csv)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
