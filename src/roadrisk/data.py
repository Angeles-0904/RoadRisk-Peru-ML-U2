import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "departamento",
    "codigo_via",
    "kilometro",
    "modalidad",
    "hora_siniestro",
    "mes",
    "dia_semana",
    "es_noche",
]

CATEGORICAL_FEATURES = ["departamento", "codigo_via", "modalidad", "dia_semana"]
NUMERIC_FEATURES = ["kilometro", "hora_siniestro", "mes", "es_noche"]
TARGET_COLUMN = "fatal"


def normalize_column(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def read_csv_latin(path: str | Path, skiprows: int = 0) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="latin1", skiprows=skiprows)


def _parse_hour(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    match = re.match(r"^(\d{1,2})", text)
    if not match:
        return np.nan
    hour = int(match.group(1))
    return float(hour if 0 <= hour <= 23 else np.nan)


def _parse_kilometer(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", ".")
    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group(0)) if match else np.nan


def load_sutran_for_model(path: str | Path) -> pd.DataFrame:
    raw = read_csv_latin(path)
    raw.columns = [normalize_column(c) for c in raw.columns]

    required = {"fecha", "hora", "departamento", "codigo_via", "kilometro", "modalidad", "fallecidos"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en SUTRAN: {sorted(missing)}")

    df = raw.copy()
    df["fecha_dt"] = pd.to_datetime(df["fecha"].astype(str), format="%Y%m%d", errors="coerce")
    df["hora_siniestro"] = df["hora"].map(_parse_hour)
    df["kilometro"] = df["kilometro"].map(_parse_kilometer)
    df["mes"] = df["fecha_dt"].dt.month.astype("float")
    df["dia_semana"] = df["fecha_dt"].dt.day_name(locale=None).fillna("desconocido")
    df["es_noche"] = df["hora_siniestro"].between(18, 23).fillna(False).astype(int)
    df["fatal"] = (pd.to_numeric(df["fallecidos"], errors="coerce").fillna(0) > 0).astype(int)

    for col in ["departamento", "codigo_via", "modalidad", "dia_semana"]:
        df[col] = (
            df[col]
            .astype("string")
            .fillna("DESCONOCIDO")
            .str.strip()
            .str.upper()
            .replace({"": "DESCONOCIDO"})
        )

    model_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    return model_df.dropna(subset=["hora_siniestro", "mes"])


def load_onsv_summary(siniestros_path: str | Path, personas_path: str | Path) -> dict:
    siniestros = read_csv_latin(siniestros_path, skiprows=4)
    personas = read_csv_latin(personas_path, skiprows=3)
    siniestros.columns = [normalize_column(c) for c in siniestros.columns]
    personas.columns = [normalize_column(c) for c in personas.columns]

    return {
        "siniestros_fatales_rows": int(len(siniestros)),
        "personas_involucradas_rows": int(len(personas)),
        "fallecidos_onsv": int(pd.to_numeric(siniestros.get("cantidad_de_fallecidos"), errors="coerce").fillna(0).sum()),
        "lesionados_onsv": int(pd.to_numeric(siniestros.get("cantidad_de_lesionados"), errors="coerce").fillna(0).sum()),
        "departamentos_top": siniestros.get("departamento", pd.Series(dtype=str)).value_counts().head(5).to_dict(),
        "clases_top": siniestros.get("clase_siniestro", pd.Series(dtype=str)).value_counts().head(5).to_dict(),
        "personas_gravedad": personas.get("gravedad", pd.Series(dtype=str)).value_counts().to_dict(),
    }

