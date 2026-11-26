"""Selección final, comparación de proyectos y simulación modular."""
import json
from types import SimpleNamespace
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from app.common import model_panel, show_figure, simulator
from src.data import ROOT, fingerprints


def consolidated():
    data = json.loads((ROOT / "outputs/consolidated.json").read_text(encoding="utf-8"))
    if data["sha256"] != fingerprints():
        raise ValueError("Los datos cambiaron respecto a la comparación guardada.")
    return data


@st.cache_resource(show_spinner="Cargando pipeline guardado…")
def load_pipeline(path, stamp):
    return joblib.load(ROOT / "outputs" / path)["pipeline"]


def winner_panel(data, compact=False):
    winner = next(m for m in data["models"] if m["id"] == data["winner_id"])
    st.markdown(f'''<div class="winner-panel"><div class="eyebrow">MEJOR MODELO · COMPARACIÓN DE TRES VERSIONES</div>
<h2>k-NN con geometría de centroides</h2><p>La mayor media de F1 macro con decisión estándar entre las 116 configuraciones revisadas.</p>
<div class="score">{winner['f1_macro']:.4f}<span> F1 macro · validación cruzada</span></div>
<div class="chips"><span>23 vecinos</span><span>Coseno</span><span>Votos 1/distancia</span><span>Centroides 2 + 6</span><span>Sin PCA</span></div></div>''', unsafe_allow_html=True)
    if not compact:
        columns = st.columns(4)
        columns[0].metric("F1 macro · CV", f"{winner['f1_macro']:.4f}")
        columns[1].metric("Mejora sobre versión base", f"+{winner['f1_macro']-data['previous_winner_f1']:.4f}")
        columns[2].metric("Configuraciones revisadas", str(data['total_configurations']))
        columns[3].metric("Variables de entrada al k-NN", "40")
        st.subheader("Los parámetros que hacen este resultado reproducible")
        left, right = st.columns([1, 1.25])
        with left:
            st.table(pd.DataFrame({"Parámetro": ["Vecinos k", "Disimilitud", "Votación", "Clústeres geométricos", "Semilla", "PCA", "Regla de decisión"],
                                   "Valor": ["23", "coseno", "ponderada: 1/distancia", "2 y 6", "42", "No", "predict estándar (mayor voto)"]}))
        with right:
            st.markdown("**Pipeline completo, ya guardado**")
            st.write(winner['preprocessing'])
            st.write("Los seis atributos derivados incluyen SO₂ ligado y varios cocientes; cinco logaritmos estabilizan variables asimétricas. Las distancias euclidianas y coseno a centroides amplían de 22 a 40 coordenadas.")
            st.caption("El modelo y sus transformaciones se cargan de disco. Para demostrar una predicción no se repite el entrenamiento.")
        st.info("Ganador por validación disponible, no por test oculto. La diferencia entre medias no demuestra superioridad estadística; las repeticiones de perfiles siguen siendo una limitación.")
    return winner


def comparison(data):
    st.header("Comparativa de modelos")
    st.write("Se reutilizaron resultados de las tres versiones con los mismos datos y cinco folds estratificados, semilla 42. No se ejecutó una nueva búsqueda.")
    frame = pd.DataFrame(data['models'])
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    bars = ax.barh(frame.name[::-1], frame.f1_macro[::-1], color=["#a78bfa"] * (len(frame)-1) + ["#f43f5e"])
    ax.bar_label(bars, fmt="%.4f", padding=6)
    ax.set(xlim=(0, 1), xlabel="F1 macro · decisión estándar", title="Representantes seleccionados para el simulador")
    show_figure(fig)
    st.dataframe(frame[['name', 'source', 'f1_macro', 'std']].rename(columns={
        "name": "Modelo", "source": "Procedencia", "f1_macro": "F1 macro", "std": "Desv. muestral"}), hide_index=True, width="stretch")
    st.caption("Una desviación vacía indica que la fuente no guardó la dispersión de la decisión estándar. No se sustituye por la del umbral ajustado.")
    st.subheader("El mejor resultado de cada proyecto")
    sources = pd.DataFrame(data['source_winners'])
    st.dataframe(sources[['source','name','f1_macro']].rename(columns={"source":"Proyecto", "name":"Mejor configuración", "f1_macro":"F1 macro estándar"}), hide_index=True, width="stretch")
    with st.expander("Ver las 116 configuraciones y sus resultados reportados"):
        all_results = pd.read_csv(ROOT / "outputs/project_comparison.csv")
        st.dataframe(all_results.rename(columns={"source":"Proyecto", "name":"Configuración", "family":"Familia", "f1_macro":"F1 estándar (ranking)", "reported_f1":"F1 reportado en origen", "adjusted_threshold":"Origen ajustó umbral", "std":"Desv. estándar"}), hide_index=True, width="stretch")
        st.download_button("Descargar comparación", all_results.to_csv(index=False), "project_comparison.csv", "text/csv")
    st.caption(data['threshold_note'])
    st.caption("MLP conserva el mejor resultado del proyecto base. SVM RBF y polinomial reutilizan pipelines de Alejo con decisión estándar. No se mezclaron predicciones mediante un ensamble sin validar.")


def modular_simulation(data, train, meta, results, specs, method):
    st.header("Simulador modular")
    models = data['models']
    preferred = {"k-NN":"selected_knn", "k-means":"selected_kmeans", "SVM lineal":"selected_svm_linear",
                 "SVM polinomial":"selected_svm_poly", "SVM RBF":"selected_svm_rbf", "MLP":"selected_mlp"}
    if method == "PCA":
        st.write("PCA transforma las coordenadas; necesita un clasificador para predecir la clase.")
        model_panel([s for s in specs if s.pca], results, train, meta, "pca_demo")
        return
    identifiers = [m['id'] for m in models]
    choice = st.selectbox("Modelo que quieres simular", identifiers,
                          index=identifiers.index(preferred.get(method, data['winner_id'])),
                          format_func=lambda value: next(m['name'] for m in models if m['id']==value), key="selected_model")
    config = next(m for m in models if m['id'] == choice)
    if choice == data['winner_id']:
        st.success("Mejor modelo disponible en la comparación · listo para predecir")
    a, b = st.columns([1, 2])
    a.metric("F1 macro · CV estándar", f"{config['f1_macro']:.4f}")
    b.caption(config['preprocessing'])
    with st.expander("Parámetros exactos del modelo seleccionado", expanded=True):
        st.json(config['params'])
    loader = lambda: (load_pipeline(config['artifact'], (ROOT / "outputs" / config['artifact']).stat().st_mtime_ns), [])
    simulator(SimpleNamespace(family=config['family']), train, meta, "consolidated_" + choice, pipeline_loader=loader)
    with st.expander("Explorar otras distancias y configuraciones del proyecto base"):
        st.caption("Estas alternativas conservan sus métricas originales. No cambian el ganador ni el CSV de entrega.")
        model_panel([s for s in specs if s.family == config['family']], results, train, meta, "extra_"+choice)
