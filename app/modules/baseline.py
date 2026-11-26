import streamlit as st
from app.common import metrics_table, model_panel


def render(train, test, results, meta, specs, method):
    st.header("3 · Baseline por distancias")
    choices = ["k-NN", "k-means"]
    family = st.selectbox("Método base", choices, index=int(method == "k-means"))
    if family == "k-NN":
        st.write("k-NN busca los k vinos más cercanos y vota con sus etiquetas. El baseline compara k=3, 5 y 7 con Euclidiana y Manhattan. También puedes explorar Minkowski y coseno en las configuraciones ampliadas.")
        st.latex(r"d_p(x,z)=\left(\sum_j |x_j-z_j|^p\right)^{1/p}")
        st.caption("p=1: Manhattan; p=2: Euclidiana. 1−coseno es una disimilitud: no cumple todos los axiomas métricos.")
    else:
        st.write("k-means aprende centroides sin etiquetas. Después se asigna a cada clúster la clase mayoritaria usando solo el entrenamiento de cada fold. El número del clúster no es la clase.")
        st.latex(r"J=\sum_{k=1}^{K}\sum_{x_i\in S_k}\|x_i-\mu_k\|_2^2")
    if not results.empty:
        metrics_table(results[(results.stage == 3) & (results.family == family)])
    options = [s for s in specs if s.family == family and (family == "k-NN" or s.stage == 3)]
    model_panel(options, results, train, meta, "baseline_" + family)
