from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SUTRAN_CSV = RAW_DIR / "sutran_accidentes_2020_2021.csv"
ONSV_SINIESTROS_CSV = RAW_DIR / "onsv_siniestros_fatales_2021_2025.csv"
ONSV_PERSONAS_CSV = RAW_DIR / "onsv_personas_involucradas_2021_2025.csv"

MODEL_PATH = MODEL_DIR / "roadrisk_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

