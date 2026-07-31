from src.roadrisk.config import SUTRAN_CSV
from src.roadrisk.data import FEATURE_COLUMNS, TARGET_COLUMN, load_sutran_for_model


def test_sutran_loader_has_features_and_target():
    df = load_sutran_for_model(SUTRAN_CSV)
    assert len(df) > 1000
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(df.columns)
    assert df[TARGET_COLUMN].nunique() == 2

