"""Entrenamiento del modelo no supervisado (K-Means) - punto de entrada CLI.

Uso:
    python -m src.roadrisk.train_clustering [--sutran-csv RUTA] [--no-promote]
"""

import argparse
import json

from .clustering import train_clustering
from .config import SUTRAN_CSV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sutran-csv", default=str(SUTRAN_CSV))
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Registra la version sin promoverla a produccion",
    )
    args = parser.parse_args()
    metrics = train_clustering(args.sutran_csv, promote=not args.no_promote)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
