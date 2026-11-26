"""Selección final, comparación de proyectos y simulación modular."""
import json
from types import SimpleNamespace
import joblib
import pandas as pd
import streamlit as st
from app.common import model_panel, simulator
from src.data import ROOT, fingerprints
from app.charts import ranking_frame, ranking_figure


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
        columns = st.columns(5)
        columns[0].metric("F1 macro · CV", f"{winner['f1_macro']:.4f}")
        columns[1].metric("Mejora sobre versión base", f"+{winner['f1_macro']-data['previous_winner_f1']:.4f}")
        columns[2].metric("Configuraciones revisadas", str(data['total_configurations']))
        columns[3].metric("Variables de entrada al k-NN", "40")
        training_seconds = next((r.get('fit_seconds') for r in data.get('source_winners', [])
                                 if r['source'] == winner['source']), None)
        columns[4].metric("Entrenamiento (s/fold)", f"{training_seconds:.4f}" if training_seconds is not None else "Sin dato")
        st.caption("Tiempo medio de ajuste por fold, recuperado de la ejecución original. No es tiempo de predicción ni de carga; depende del equipo utilizado.")
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
    st.subheader("Ranking completo de modelos")
    all_results = ranking_frame()
    families = st.multiselect("Familias visibles", ['k-NN','SVM','MLP','k-means'], default=['k-NN','SVM','MLP','k-means'])
    visible = all_results[all_results.family.isin(families)]
    st.caption(f"{len(visible)} de {len(all_results)} configuraciones · ordenadas de mayor a menor F1 macro · el ganador está en rojo")
    if not visible.empty:
        left,right = st.columns([1.35, 1])
        with left:
            with st.container(height=720, border=False):
                st.plotly_chart(ranking_figure(visible), theme=None, width="stretch", key="complete_model_ranking")
        with right:
            compact_table = visible[['rank','display_name','f1_macro','fit_seconds']].copy()
            compact_table['display_name'] = '[' + visible['source'].replace({
                'Proyecto base':'Base','Versión alternativa':'Alternativa','Versión Alejo':'Alejo'}) + '] ' + visible['display_name']
            st.dataframe(compact_table.rename(columns={
                'rank':'Puesto','display_name':'Configuración','f1_macro':'F1 macro','fit_seconds':'Entrenamiento (s/fold)'}),
                hide_index=True, width="stretch", height=720, column_config={
                    'Puesto':st.column_config.NumberColumn(width='small'),
                    'Configuración':st.column_config.TextColumn(width='medium'),
                    'F1 macro':st.column_config.ProgressColumn(format='%.4f',min_value=0,max_value=1,width='small'),
                    'Entrenamiento (s/fold)':st.column_config.NumberColumn(format='%.4f',width='small')})
    else:
        st.info("Selecciona una familia para mostrar su ranking.")
    st.download_button("Descargar ranking completo", all_results.to_csv(index=False), "ranking_modelos.csv", "text/csv")
    st.caption("Desplázate dentro de la gráfica o de la tabla para ver todo el ranking. Pasa el cursor sobre una barra para ver la configuración completa. Las barras de error solo aparecen cuando la fuente guardó la desviación estándar.")
    st.caption("Entrenamiento: segundos medios por fold ya registrados en cada proyecto. Los equipos pueden ser distintos; estos tiempos no se utilizan para elegir el ganador.")
    st.subheader("El mejor resultado de cada proyecto")
    sources = pd.DataFrame(data['source_winners'])
    st.dataframe(sources[['source','name','f1_macro']].rename(columns={"source":"Proyecto", "name":"Mejor configuración", "f1_macro":"F1 macro estándar"}), hide_index=True, width="stretch")
    with st.expander("Detalle adicional: resultados con y sin ajuste de umbral"):
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
