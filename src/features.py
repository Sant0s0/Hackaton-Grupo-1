"""Transformaciones compatibles con Pipeline: nada se ajusta sobre validación."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_array, check_is_fitted

RATIO_NAMES = ["fixed_volatile_ratio", "free_total_so2_ratio", "alcohol_density_ratio"]


class ChemicalRatios(TransformerMixin, BaseEstimator):
    """11 entradas en el orden de FEATURES y tres cocientes interpretables.

    Si el denominador es casi cero, devuelve NaN para imputar con la mediana
    aprendida dentro del fold; no convierte una división inválida en un infinito.
    """
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        if self.n_features_in_ != 11:
            raise ValueError("ChemicalRatios espera las 11 variables fisicoquímicas.")
        return self

    def transform(self, X):
        check_is_fitted(self)
        a = np.asarray(X, dtype=float)
        if a.ndim != 2 or a.shape[1] != self.n_features_in_:
            raise ValueError("Número de características incorrecto.")
        ratios = []
        for numerator, denominator in ((0, 1), (5, 6), (10, 7)):
            ratio = np.full(len(a), np.nan)
            np.divide(a[:, numerator], a[:, denominator], out=ratio,
                      where=np.abs(a[:, denominator]) > 1e-9)
            ratios.append(ratio)
        return np.column_stack([a, *ratios])

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(11)]
        return np.array([*input_features, *RATIO_NAMES], dtype=object)


class CentroidDistances(TransformerMixin, BaseEstimator):
    """Amplía las coordenadas escaladas con distancia L2 a centroides del fold."""
    def __init__(self, n_clusters=8, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        a = check_array(X)
        self.n_features_in_ = a.shape[1]
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10,
                              random_state=self.random_state).fit(a)
        return self

    def transform(self, X):
        check_is_fitted(self, "kmeans_")
        a = check_array(X)
        return np.column_stack([a, self.kmeans_.transform(a)])
