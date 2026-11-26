import streamlit as st
from app.common import show_figure
from src.data import FEATURES, audit_data
from src.plots import class_figure, correlation_figure, distribution_figure


def render(train, test, results, meta, specs, method):
    st.header("2 · Exploración y limpieza")
    audit = audit_data(train, test)
    left, right = st.columns(2)
    with left:
        show_figure(class_figure(train))
    with right:
        st.subheader("Qué encontramos")
        st.write(f"Faltantes en train: **{audit['missing_train']}**. Faltantes en test: **{audit['missing_test']}**.")
        st.write(f"Filas repetidas por sus once mediciones en train: **{audit['duplicate_feature_rows_train']}**. Se conservan y se hace un control adicional por grupos.")
        st.write("id se excluye de los predictores. Se imputan medianas solo si hace falta y se estandarizan las variables dentro de cada fold.")
        st.caption("Los valores extremos se describen con IQR; una observación extrema no se elimina sin evidencia de error.")
    show_figure(correlation_figure(train))
    show_figure(distribution_figure(train))
    with st.expander("Estadísticas y banderas de valores extremos"):
        st.dataframe(train[FEATURES].describe().T, width="stretch")
        st.json(audit["iqr_flags_by_feature"])
