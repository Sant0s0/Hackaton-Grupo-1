"""Figuras compartidas entre la aplicación, el notebook y el informe."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.data import FEATURES
from src.features import ChemicalRatios

COLORS = ["#396d99", "#c05c7d"]
plt.rcParams.update({"figure.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def class_figure(train):
    counts = train.target.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    bars = ax.bar(["0 · Estándar", "1 · Superior"], counts, color=COLORS, width=0.55)
    ax.bar_label(bars, labels=[f"{n:,} · {n/len(train):.1%}" for n in counts], padding=5)
    ax.set(ylim=(0, counts.max() * 1.18), ylabel="Número de vinos", title="Cada clase cuenta igual en F1 macro")
    return fig


def correlation_figure(train):
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    corr = train[FEATURES].corr()
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=70, ha="right")
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    ax.set_title("Correlación de Pearson · predictores originales")
    fig.colorbar(image, ax=ax, shrink=0.75)
    return fig


def distribution_figure(train):
    columns = ["alcohol", "volatile_acidity", "density", "total_sulfur_dioxide"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    for ax, col in zip(axes.flat, columns):
        for label, color in enumerate(COLORS):
            ax.hist(train.loc[train.target == label, col], bins=35, density=True,
                    alpha=0.55, label=f"Clase {label}", color=color)
        ax.set(title=col, ylabel="Densidad de frecuencia")
        ax.legend(frameon=False)
    return fig


def pca_analysis(train, ratios=True):
    X = train[FEATURES]
    values = ChemicalRatios().fit_transform(X) if ratios else X.to_numpy()
    values = SimpleImputer(strategy="median").fit_transform(values)
    values = StandardScaler().fit_transform(values)
    pca = PCA(svd_solver="full").fit(values)
    projection = pca.transform(values)
    return pca, projection


def pca_figure(train, ratios=True):
    pca, projection = pca_analysis(train, ratios)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for label, color in enumerate(COLORS):
        mask = train.target.to_numpy() == label
        axes[0].scatter(projection[mask, 0], projection[mask, 1], s=8, alpha=0.35,
                        color=color, label=f"Clase {label}")
    axes[0].set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                title="Proyección descriptiva: dos componentes")
    axes[0].legend(frameon=False)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(np.arange(1, len(cumulative)+1), cumulative, "o-", color=COLORS[0])
    for threshold in (0.90, 0.95):
        axes[1].axhline(threshold, ls="--", lw=1, label=f"{threshold:.0%}")
    axes[1].set(xlabel="Número de componentes", ylabel="Varianza acumulada", ylim=(0, 1.03),
                title="La varianza retenida no equivale a F1")
    axes[1].legend(frameon=False)
    return fig


def confusion_figure(matrix):
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=16,
                color="white" if value > np.max(matrix)/2 else "#18344a")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicción fuera de fold",
           ylabel="Clase real", title="Matriz de confusión · ganador por CV")
    return fig
