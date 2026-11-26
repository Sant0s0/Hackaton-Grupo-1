import streamlit as st
from app.common import metrics_table, model_panel


def render(train, test, results, meta, specs, method):
    st.header("5 · SVM y redes MLP")
    methods = ["SVM lineal", "SVM polinomial", "SVM RBF", "MLP"]
    selected = st.selectbox("Modelo avanzado", methods, index=methods.index(method) if method in methods else 2)
    if selected == "MLP":
        st.write("Un MLP transforma las mediciones mediante capas de pesos, sesgos y activaciones no lineales. Se comparan una y dos capas, ReLU/tanh y regularización alpha; Adam actualiza los pesos a partir de backpropagation.")
        st.latex(r"h=\phi(W_1x+b_1),\quad \hat p=\sigma(W_2h+b_2)")
        st.caption("La red minimiza entropía cruzada regularizada. F1 macro se usa para seleccionar modelos, no como pérdida diferenciable.")
        options = [s for s in specs if s.family == "MLP"]
    else:
        kernel = {"SVM lineal": "linear", "SVM polinomial": "poly", "SVM RBF": "rbf"}[selected]
        st.write("SVM busca un margen amplio y penaliza violaciones con C. Los kernels permiten fronteras no lineales. Se comparan los pesos de clase para atender el desbalance.")
        st.latex(r"\min_{w,b,\xi}\frac12\|w\|^2+C\sum_i\xi_i")
        if kernel == "rbf":
            st.latex(r"K(x,z)=\exp(-\gamma\|x-z\|^2)")
        elif kernel == "poly":
            st.latex(r"K(x,z)=(\gamma x^Tz+r)^d")
        else:
            st.latex(r"K(x,z)=x^Tz")
        options = [s for s in specs if s.family == "SVM" and s.params["kernel"] == kernel]
    if not results.empty:
        metrics_table(results[results.name.isin([s.name for s in options])])
    model_panel(options, results, train, meta, "advanced_" + selected)
