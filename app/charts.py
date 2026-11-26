"""Gráficas interactivas del ranking y PCA, visibles sin desplegables."""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.data import FEATURES, ROOT
from src.pipeline_wine import PhysicochemicalFeatures

FAMILY_COLORS = {"k-NN": "#38bdf8", "SVM": "#a78bfa", "MLP": "#f472b6", "k-means": "#94a3b8"}
PROJECT_NAMES = {"Proyecto base": "Base", "Versión alternativa": "Alternativa", "Versión Alejo": "Alejo"}


def style_figure(fig, height):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(255,255,255,.025)", font=dict(family="Segoe UI, sans-serif", color="#e2e8f0", size=13),
                      margin=dict(l=20, r=45, t=30, b=55), hoverlabel=dict(font_size=14))
    return fig


def ranking_frame():
    frame = pd.read_csv(ROOT / "outputs/project_comparison.csv").sort_values(
        ["f1_macro", "source", "id"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
    specs = {s['name']: s for s in json.loads((ROOT / "outputs/candidates.json").read_text(encoding="utf-8"))}
    distances = {"euclidean":"Euclidiana", "manhattan":"Manhattan", "minkowski":"Minkowski", "cosine":"coseno"}
    def friendly(row):
        if row['source'] != "Proyecto base":
            return row['name']
        spec = specs[row['id']]
        p = spec['params']
        if spec['family'] == "k-NN":
            label = f"k-NN k={p['n_neighbors']} · {distances.get(p.get('metric'), p.get('metric'))} · " + ("1/d" if p.get('weights') == 'distance' else "uniforme")
            if p.get('metric') == 'minkowski': label += f" p={p.get('p', 2)}"
        elif spec['family'] == "SVM":
            label = f"SVM {p['kernel']} · C={p['C']}"
            if p['kernel'] != 'linear': label += f" · γ={p.get('gamma', 'scale')}"
            if p.get('class_weight'): label += " · balanceada"
        elif spec['family'] == "MLP":
            label = f"MLP {tuple(p['hidden_layer_sizes'])} · {p['activation']} · α={p['alpha']}"
            if p.get('balanced'): label += " · balanceada"
        else:
            label = f"k-means K={p['n_clusters']} · mayoría"
        if spec['ratios']: label += " + ratios"
        if spec['pca']: label += f" + PCA {spec['pca']:.0%}"
        if spec['centroids']: label += f" + {spec['centroids']} centroides"
        return label
    frame['display_name'] = frame.apply(friendly, axis=1)
    frame.insert(0, 'rank', np.arange(1, len(frame)+1))
    return frame


def ranking_figure(frame):
    labels = [f"{r['rank']:03d} · {r['display_name'][:64]}{'…' if len(r['display_name'])>64 else ''} · {PROJECT_NAMES[r['source']]}"
              for _, r in frame.iterrows()]
    hover = [[r['display_name'], r['source'], r['family'],
              "No disponible" if pd.isna(r['std']) else f"{r['std']:.4f}"] for _, r in frame.iterrows()]
    fig = go.Figure(go.Bar(x=frame.f1_macro.tolist(), y=labels, orientation='h',
        marker_color=["#f43f5e" if r['rank'] == 1 else FAMILY_COLORS[r['family']] for _, r in frame.iterrows()],
        text=[f"{v:.4f}" for v in frame.f1_macro], textposition="outside", cliponaxis=False,
        error_x=dict(type="data", array=frame['std'].fillna(0).tolist(), color="#e2e8f0", thickness=1, width=3),
        customdata=hover, hovertemplate="<b>%{customdata[0]}</b><br>Proyecto: %{customdata[1]}<br>Familia: %{customdata[2]}<br>F1 macro estándar: %{x:.4f}<br>Desv. CV: %{customdata[3]}<extra></extra>"))
    style_figure(fig, max(470, len(frame)*32+90))
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=labels, automargin=True, tickfont_size=12)
    fig.update_xaxes(title="F1 macro · decisión estándar (eje ampliado)", range=[.55, .93], gridcolor="rgba(255,255,255,.08)")
    fig.update_layout(margin=dict(l=15, r=55, t=15, b=50), bargap=.25)
    return fig


@st.cache_data(show_spinner=False)
def pca_data(train, engineered=False):
    values = PhysicochemicalFeatures().transform(train[FEATURES]) if engineered else train[FEATURES].to_numpy()
    scaled = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(values))
    pca = PCA(svd_solver="full").fit(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    table = pd.DataFrame({"Componente":[f"PC{i+1}" for i in range(len(cumulative))],
                          "Autovalor λ":pca.explained_variance_,
                          "Varianza individual (%)":pca.explained_variance_ratio_*100,
                          "Varianza acumulada (%)":cumulative*100})
    n90 = int(np.searchsorted(cumulative, .90)+1)
    n95 = int(np.searchsorted(cumulative, .95)+1)
    return table, n90, n95


def pca_spectrum_figure(table, n90, n95):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table.Componente, y=table['Varianza individual (%)'], name="Varianza individual (%)",
                         marker_color="#38bdf8", opacity=.85, customdata=table['Autovalor λ'],
                         hovertemplate="%{x}<br>Autovalor λ: %{customdata:.4f}<br>Varianza individual: %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=table.Componente, y=table['Varianza acumulada (%)'], mode="lines+markers",
                            name="Varianza acumulada (%)", line=dict(color="#f43f5e", width=3), marker=dict(size=8),
                            hovertemplate="%{x}<br>Varianza acumulada: %{y:.2f}%<extra></extra>"))
    fig.add_hline(y=90, line_dash="dash", line_color="#34d399", annotation_text=f"90% · {n90} componentes",
                  annotation_position="bottom right", annotation_font_color="#34d399")
    fig.add_hline(y=95, line_dash="dot", line_color="#facc15", annotation_text=f"95% · {n95} componentes",
                  annotation_position="top right", annotation_font_color="#facc15")
    style_figure(fig, 470)
    fig.update_layout(legend=dict(orientation="h", y=1.16, x=0), margin=dict(l=50,r=20,t=75,b=55))
    fig.update_xaxes(title="Componentes principales", tickangle=0)
    fig.update_yaxes(title="Porcentaje de varianza explicada (%)", range=[0,105], gridcolor="rgba(255,255,255,.08)")
    return fig


def render_pca_overview(train, key="main"):
    st.subheader("Espectro de autovalores y varianza acumulada PCA")
    representation = st.radio("Representación para PCA", ["11 mediciones originales", "22 atributos químicos de Alejo"],
                              horizontal=True, key=f"pca_representation_{key}")
    table, n90, n95 = pca_data(train, representation.startswith('22'))
    a,b,c = st.columns(3)
    a.metric("Para conservar el 90%", f"{n90} componentes")
    b.metric("Para conservar el 95%", f"{n95} componentes")
    c.metric("Variables analizadas", str(len(table)))
    left, right = st.columns([1.6, 1])
    with left:
        st.plotly_chart(pca_spectrum_figure(table,n90,n95), theme=None, width="stretch", key=f"pca_spectrum_{key}")
    with right:
        st.dataframe(table, hide_index=True, height=470, width="stretch", column_config={
            "Autovalor λ":st.column_config.NumberColumn(format="%.3f"),
            "Varianza individual (%)":st.column_config.NumberColumn(format="%.2f %%"),
            "Varianza acumulada (%)":st.column_config.ProgressColumn(format="%.2f %%", min_value=0, max_value=100)})
    st.caption("Barras cian: autovalores normalizados como porcentaje de varianza. Línea vino: acumulado. λ es el autovalor sin normalizar. Análisis descriptivo sobre train; el ganador no usa PCA.")
