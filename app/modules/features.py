import streamlit as st
from app.common import metrics_table, model_panel, show_figure
from src.plots import pca_figure


def render(train, test, results, meta, specs, method):
    st.header("4 · Variables nuevas y PCA")
    st.table({"Variable nueva": ["fixed_acidity / volatile_acidity", "free_sulfur_dioxide / total_sulfur_dioxide", "alcohol / density"],
              "Idea": ["Balance relativo entre acidez fija y volátil", "Fracción del SO₂ que está libre", "Relación empírica entre grado alcohólico y densidad"]})
    st.write("Son hipótesis de representación, no leyes que determinen la calidad. Se conservan las once entradas originales. Los cocientes se calculan antes del escalado; los denominadores casi nulos se imputan dentro del fold.")
    st.latex(r"Z=(X-\mu)/\sigma,\quad \Sigma=\frac{Z^T Z}{N-1},\quad T=ZV")
    st.write("PCA gira el espacio hacia direcciones ortogonales de máxima varianza. Se prueban retenciones del 90% y 95%, y se decide con F1 macro y tiempo.")
    show_figure(pca_figure(train))
    st.caption("Esta figura es descriptiva sobre train. Los PCA usados para evaluar se ajustan de nuevo en cada fold. Dos ejes no muestran toda la información retenida por el modelo.")
    if not results.empty:
        names = ["base_knn_k5_manhattan", "fe_knn_k5_manhattan", "pca0.9_knn_k5_manhattan", "pca0.95_knn_k5_manhattan"]
        st.subheader("Comparación controlada: cambia la representación")
        metrics_table(results[results.name.isin(names)])
    st.subheader("Probar un clasificador con PCA")
    model_panel([s for s in specs if s.pca], results, train, meta, "pca")
