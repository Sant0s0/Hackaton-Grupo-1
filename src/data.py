"""Contrato único de datos para notebook, entrenamiento y aplicación."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol",
]
LABELS = {0: "Calidad estándar", 1: "Calidad superior"}


def validate_frame(frame, training=False):
    expected = ["id", *FEATURES] + (["target"] if training else [])
    if set(frame.columns) != set(expected) or len(frame.columns) != len(expected):
        raise ValueError(f"Columnas esperadas: {expected}")
    if frame.empty or frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("Los id deben ser únicos, no nulos, y el archivo no puede estar vacío.")
    numeric = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Las características no pueden contener infinitos.")
    if numeric.isna().all().any():
        raise ValueError("Una característica está completamente vacía.")
    if training and (frame.target.isna().any() or set(frame.target.unique()) != {0, 1}):
        raise ValueError("target debe contener ambas clases binarias 0 y 1, sin nulos.")
    result = frame.loc[:, expected].copy()
    result[FEATURES] = numeric
    return result


def load_data():
    train = validate_frame(pd.read_csv(ROOT / "data/train_public.csv"), training=True)
    test = validate_frame(pd.read_csv(ROOT / "data/test_public_features.csv"))
    if set(train.id) & set(test.id):
        raise ValueError("Hay identificadores compartidos entre entrenamiento y prueba.")
    return train, test


def fingerprints():
    return {name: hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest()
            for name in ("train_public.csv", "test_public_features.csv")}


def validate_submission(submission, test):
    if list(submission.columns) != ["id", "prediction"]:
        raise ValueError("submission_Grupo_1.csv requiere exactamente id,prediction, en ese orden.")
    if len(submission) != len(test) or not submission.id.equals(test.id.reset_index(drop=True)):
        raise ValueError("Las filas y el orden de id deben coincidir con test_public_features.csv.")
    if submission.isna().any().any() or not submission.prediction.isin([0, 1]).all():
        raise ValueError("Las predicciones deben ser 0 o 1, sin nulos.")
    return True


def audit_data(train, test):
    x = train[FEATURES]
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    # Regla descriptiva: una bandera IQR no demuestra un error de laboratorio.
    outliers = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()
    train_keys = pd.util.hash_pandas_object(x, index=False)
    test_keys = pd.util.hash_pandas_object(test[FEATURES], index=False)
    labels_per_profile = train.assign(profile=train_keys).groupby("profile").target.nunique()
    return {
        "train_rows": len(train), "test_rows": len(test), "features": len(FEATURES),
        "class_counts": {str(k): int(v) for k, v in train.target.value_counts().sort_index().items()},
        "missing_train": int(x.isna().sum().sum()), "missing_test": int(test[FEATURES].isna().sum().sum()),
        "duplicate_feature_rows_train": int(train_keys.duplicated().sum()),
        "test_rows_with_profile_in_train": int(test_keys.isin(train_keys).sum()),
        "profiles_with_conflicting_labels": int((labels_per_profile > 1).sum()),
        "iqr_flags_by_feature": {k: int(v) for k, v in outliers.items()},
        "sha256": fingerprints(),
    }
