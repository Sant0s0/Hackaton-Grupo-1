"""Transformaciones reutilizadas de la versión Alejo; sin ejecutar su entrenamiento.

CalibratedThresholdModel se conserva solo para leer sus archivos originales.
La consolidación extrae el pipeline aprendido y usa predict con decisión estándar.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from src.data import FEATURES as RAW_FEATURES

FEATURE_METADATA = {name: {"default": 1.0} for name in RAW_FEATURES}

class PhysicochemicalFeatures(BaseEstimator, TransformerMixin):
    """
    Transformador determinista fila-a-fila (sin estado estadístico entre muestras)
    que construye razones fisicoquímicas enológicas y estabiliza colas derechas
    con log(1 + x) para evitar que outliers distorsionen distancias y gradientes.
    """

    LOG_COLS = [
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "volatile_acidity",
        "sulphates",
    ]

    def __init__(self, use_log: bool = True):
        self.use_log = use_log

    def fit(self, X, y=None):
        return self

    def transform_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X[RAW_FEATURES].copy()
        else:
            df = pd.DataFrame(np.asarray(X, dtype=float), columns=RAW_FEATURES)

        eps = 1e-6
        # 1. Dióxido de azufre ligado (combinado con antocianos/aldehídos)
        df["bound_so2"] = np.maximum(
            df["total_sulfur_dioxide"] - df["free_sulfur_dioxide"], 0.0
        )
        # 2. Proporción de SO2 molecular libre activo
        df["free_so2_ratio"] = df["free_sulfur_dioxide"] / (
            df["total_sulfur_dioxide"] + eps
        )
        # 3. Balance de acidez estructural vs. acidez acética volátil
        df["acidity_balance"] = df["fixed_acidity"] / (df["volatile_acidity"] + eps)
        # 4. Eficiencia fermentativa alcohol / densidad
        df["alcohol_density_ratio"] = df["alcohol"] / (df["density"] + eps)
        # 5. Equilibrio organoléptico dulzor / acidez fresca
        df["sugar_acidity_ratio"] = df["residual_sugar"] / (
            df["fixed_acidity"] + df["citric_acid"] + eps
        )
        # 6. Relación mineral sulfatos / cloruros
        df["sulphates_chlorides_ratio"] = df["sulphates"] / (df["chlorides"] + eps)

        if self.use_log:
            for col in self.LOG_COLS:
                df[f"log_{col}"] = np.log1p(np.maximum(df[col], 0.0))

        return df

    def transform(self, X) -> np.ndarray:
        return self.transform_dataframe(X).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        dummy = pd.DataFrame([ {k: v["default"] for k, v in FEATURE_METADATA.items()} ])
        return list(self.transform_dataframe(dummy).columns)

class KMeansCentroidDistances(BaseEstimator, TransformerMixin):
    """
    Transformador geométrico no supervisado que ajusta k-means EXCLUSIVAMENTE
    sobre el subconjunto de entrenamiento (ya escalado) y agrega:
        - Distancias Euclidianas a los k centroides (ej. k=2 tinto/blanco, k=6 estilos)
        - Distancia al centroide más cercano (anomalía/tipicidad química)
        - Distancias Coseno a los k centroides
    Seguido de una re-estandarización interna ajustada sólo en train.
    """

    def __init__(self, n_clusters_list: Tuple[int, ...] = (2, 6), random_state: int = 42):
        self.n_clusters_list = n_clusters_list
        self.random_state = random_state

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.kmeans_models_: List[KMeans] = []
        for k in self.n_clusters_list:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            km.fit(X_arr)
            self.kmeans_models_.append(km)
        augmented = self._augment(X_arr)
        self.scaler_ = StandardScaler().fit(augmented)
        return self

    def _augment(self, X_arr: np.ndarray) -> np.ndarray:
        blocks = [X_arr]
        for km in self.kmeans_models_:
            d_euc = km.transform(X_arr)
            d_min = d_euc.min(axis=1, keepdims=True)
            d_cos = cosine_distances(X_arr, km.cluster_centers_)
            blocks.extend([d_euc, d_min, d_cos])
        return np.hstack(blocks)

    def transform(self, X) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return self.scaler_.transform(self._augment(X_arr))

class KMeansClusterClassifier(BaseEstimator, ClassifierMixin):
    """
    Baseline geométrico basado en k-means (Paso 3 de la Guía):
    Agrupa las muestras en k clústeres usando distancia Euclidiana sobre el
    espacio escalado y asigna a cada clúster su proporción empírica de clase 1
    observada en el fold de entrenamiento.
    """

    def __init__(self, n_clusters: int = 20, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.random_state
        )
        clusters = self.kmeans_.fit_predict(X_arr)
        global_prior = float(np.mean(y_arr))
        self.cluster_probs_ = np.full(self.n_clusters, global_prior, dtype=float)
        for c in range(self.n_clusters):
            mask = clusters == c
            if np.any(mask):
                self.cluster_probs_[c] = float(np.mean(y_arr[mask]))
        # Mayoría estricta (>= 0.5) por clúster
        self.cluster_labels_ = (self.cluster_probs_ >= 0.5).astype(int)
        return self

    def predict_proba(self, X) -> np.ndarray:
        clusters = self.kmeans_.predict(np.asarray(X, dtype=float))
        p1 = self.cluster_probs_[clusters]
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        clusters = self.kmeans_.predict(np.asarray(X, dtype=float))
        return self.cluster_labels_[clusters]

def extract_continuous_score(estimator, X) -> np.ndarray:
    """Extrae P(y=1|x) si existe predict_proba, o f(x) si usa decision_function."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X).astype(float)

class CalibratedThresholdModel(BaseEstimator, ClassifierMixin):
    """
    Empaqueta un Pipeline ya entrenado junto con su umbral óptimo τ* hallado
    en la versión de origen. La consolidación usa el pipeline sin ajustar umbral.
    """

    def __init__(self, pipeline: Pipeline, threshold: float = 0.5, score_kind: str = "proba"):
        self.pipeline = pipeline
        self.threshold = threshold
        self.score_kind = score_kind

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.pipeline_ = clone(self.pipeline).fit(X, y)
        return self

    def decision_scores(self, X) -> np.ndarray:
        est = getattr(self, "pipeline_", self.pipeline)
        return extract_continuous_score(est, X)

    def predict(self, X) -> np.ndarray:
        return (self.decision_scores(X) >= self.threshold).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        est = getattr(self, "pipeline_", self.pipeline)
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)
        # Aproximación logística monótona centrada en el umbral para visualización
        s = self.decision_scores(X) - self.threshold
        p1 = 1.0 / (1.0 + np.exp(-np.clip(s, -20.0, 20.0)))
        return np.column_stack([1.0 - p1, p1])
