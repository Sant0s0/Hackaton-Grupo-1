"""Registro explícito de los únicos métodos permitidos por la guía."""
from dataclasses import asdict, dataclass, field
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from src.data import SEED
from src.features import ChemicalRatios, CentroidDistances


class KMeansMajority(ClassifierMixin, BaseEstimator):
    """k-means no supervisado + correspondencia clúster/clase aprendida en train.

    Un índice de clúster no es una etiqueta. Los empates se resuelven eligiendo
    la clase menor; clústeres vacíos usan la distribución global de entrenamiento.
    """
    def __init__(self, n_clusters=8, random_state=SEED):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        clusters = self.kmeans_.fit_predict(X)
        global_counts = np.bincount(encoded, minlength=len(self.classes_))
        self.cluster_proba_ = np.empty((self.n_clusters, len(self.classes_)))
        for cluster in range(self.n_clusters):
            counts = np.bincount(encoded[clusters == cluster], minlength=len(self.classes_))
            if counts.sum() == 0:
                counts = global_counts
            self.cluster_proba_[cluster] = counts / counts.sum()
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "cluster_proba_")
        return self.cluster_proba_[self.kmeans_.predict(check_array(X))]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class CourseMLP(ClassifierMixin, BaseEstimator):
    """MLP pequeño; pesos de clase calculados únicamente con y de entrenamiento."""
    def __init__(self, hidden_layer_sizes=(32,), activation="relu", alpha=0.001,
                 balanced=False, max_iter=500, random_state=SEED):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.balanced = balanced
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes), activation=self.activation,
            alpha=self.alpha, solver="adam", learning_rate_init=0.001,
            max_iter=self.max_iter, random_state=self.random_state,
            early_stopping=False, n_iter_no_change=30,
        )
        weights = compute_sample_weight("balanced", y) if self.balanced else None
        self.model_.fit(X, y, sample_weight=weights)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)


@dataclass(frozen=True)
class Candidate:
    name: str
    stage: int
    family: str
    params: dict = field(default_factory=dict)
    ratios: bool = False
    pca: float | None = None
    centroids: int | None = None

    def to_dict(self):
        return asdict(self)


def build_pipeline(spec):
    if isinstance(spec, dict):
        spec = Candidate(**spec)
    factories = {"k-NN": KNeighborsClassifier, "k-means": KMeansMajority,
                 "SVM": SVC, "MLP": CourseMLP}
    if spec.family not in factories:
        raise ValueError("Familia de modelos no permitida.")
    steps = []
    if spec.ratios:
        steps.append(("ratios", ChemicalRatios()))
    steps.extend([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    if spec.centroids:
        steps.extend([("centroids", CentroidDistances(spec.centroids)), ("rescale", StandardScaler())])
    if spec.pca:
        if not 0.90 <= spec.pca <= 0.99:
            raise ValueError("La retención de varianza debe estar entre 0.90 y 0.99.")
        steps.append(("pca", PCA(n_components=spec.pca, svd_solver="full")))
    params = dict(spec.params)
    if spec.family == "SVM":
        params.setdefault("cache_size", 256)
        params.setdefault("random_state", SEED)
    if spec.family == "k-NN":
        params.setdefault("algorithm", "brute")
        params.setdefault("n_jobs", 1)
    steps.append(("model", factories[spec.family](**params)))
    return Pipeline(steps)


def candidates(profile="complete"):
    """Comparaciones predeterminadas y auditables, sin AutoML ni modelos ajenos."""
    specs = []
    for k in (3, 5, 7):
        for metric in ("euclidean", "manhattan"):
            specs.append(Candidate(f"base_knn_k{k}_{metric}", 3, "k-NN",
                                   {"n_neighbors": k, "metric": metric}))
    for k in (2, 16):
        specs.append(Candidate(f"base_kmeans_{k}", 3, "k-means", {"n_clusters": k}))
    # Ablación: la misma configuración con/sin ratios y PCA.
    specs.append(Candidate("fe_knn_k5_manhattan", 4, "k-NN",
                           {"n_neighbors": 5, "metric": "manhattan"}, ratios=True))
    for variance in (0.90, 0.95):
        specs.append(Candidate(f"pca{variance}_knn_k5_manhattan", 4, "k-NN",
                               {"n_neighbors": 5, "metric": "manhattan"}, ratios=True, pca=variance))
    for metric in ("euclidean", "manhattan", "minkowski", "cosine"):
        for k in ((3, 5, 7, 11) if profile == "complete" else (5,)):
            params = {"n_neighbors": k, "metric": metric, "weights": "distance"}
            if metric == "minkowski":
                params["p"] = 3
            specs.append(Candidate(f"fe_knn_k{k}_{metric}_distance", 4, "k-NN", params, ratios=True))
    specs.append(Candidate("centroids8_knn", 4, "k-NN",
                           {"n_neighbors": 5, "weights": "distance", "metric": "manhattan"},
                           ratios=True, centroids=8))
    for kernel in ("linear", "poly", "rbf"):
        c_values = (0.1, 1, 10) if kernel == "linear" else (1, 10, 100)
        if profile != "complete":
            c_values = (1,)
        for c in c_values:
            for weight in (None, "balanced"):
                gammas = ("scale", 0.03, 0.1) if kernel == "rbf" and profile == "complete" else ("scale",)
                for gamma in gammas:
                    params = {"kernel": kernel, "C": c, "gamma": gamma, "class_weight": weight}
                    if kernel == "poly":
                        params["degree"] = 3
                    specs.append(Candidate(f"svm_{kernel}_C{c}_g{gamma}_w{weight}", 5, "SVM", params, ratios=True))
    for variance in (0.90, 0.95):
        specs.append(Candidate(f"pca{variance}_svm_rbf", 5, "SVM",
                               {"kernel": "rbf", "C": 10, "gamma": "scale", "class_weight": "balanced"},
                               ratios=True, pca=variance))
    specs.append(Candidate("centroids8_svm_rbf", 5, "SVM",
                           {"kernel": "rbf", "C": 10, "gamma": "scale", "class_weight": "balanced"},
                           ratios=True, centroids=8))
    for layers in ((32,), (64, 32)) if profile == "complete" else ((32,),):
        for activation in ("relu", "tanh"):
            for alpha in (0.0001, 0.01) if profile == "complete" else (0.01,):
                for balanced in (False, True):
                    params = {"hidden_layer_sizes": layers, "activation": activation,
                              "alpha": alpha, "balanced": balanced}
                    name = f"mlp_{'_'.join(map(str,layers))}_{activation}_a{alpha}_b{balanced}"
                    specs.append(Candidate(name, 5, "MLP", params, ratios=True))
    for activation in ("relu", "tanh"):
        specs.append(Candidate(f"pca0.95_mlp_{activation}", 5, "MLP",
                               {"hidden_layer_sizes": (32,), "activation": activation,
                                "alpha": 0.01, "balanced": True}, ratios=True, pca=0.95))
    return specs
