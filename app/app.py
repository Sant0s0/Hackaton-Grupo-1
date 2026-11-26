"""Aplicación consolidada: ganador, comparación y simulador modular."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from app.common import saved_experiment, show_figure
from app.dashboard import consolidated, winner_panel, comparison, modular_simulation
from app.modules import setup, eda, baseline, features, advanced, delivery
from src.data import ROOT, load_data, validate_submission
from src.plots import class_figure
from app.charts import render_pca_overview
import pandas as pd

st.set_page_config(page_title="WineLab · Comparación de modelos", page_icon="🍷", layout="wide")
st.session_state['dark_visuals'] = True
st.markdown("""<style>
.stApp{background:radial-gradient(ellipse at 10% 0%,#321128 0%,#0b1020 50%,#060a14 100%)}
.block-container{max-width:1250px;padding-top:1.8rem}h1,h2,h3{letter-spacing:-.025em;color:#f8fafc}
[data-testid="stSidebar"]{background:rgba(10,14,30,.96);border-right:1px solid #253047}
[data-testid="stMetric"]{background:rgba(255,255,255,.045);border:1px solid #303750;border-radius:15px;padding:17px}
.brand{padding-bottom:20px}.brand h1{font-size:2.7rem;margin:0;background:linear-gradient(90deg,#fda4af,#f43f5e,#a78bfa,#38bdf8);background-clip:text;-webkit-text-fill-color:transparent}
.brand p{color:#b4c2d6;margin:.4rem 0}.winner-panel{padding:28px 32px;margin:10px 0 24px;border-radius:22px;border:1px solid rgba(244,63,94,.38);background:linear-gradient(120deg,rgba(244,63,94,.17),rgba(167,139,250,.13),rgba(56,189,248,.07))}
.winner-panel h2{font-size:2rem;margin:.6rem 0}.winner-panel p{color:#cbd5e1}.eyebrow{font-size:.8rem;letter-spacing:.13em;color:#fda4af;font-weight:700}.score{font-size:3.4rem;font-weight:800;color:#fff}.score span{font-size:1rem;font-weight:400;color:#cbd5e1}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.chips span{padding:6px 12px;border-radius:24px;background:rgba(255,255,255,.07);border:1px solid #4b4561;font-size:.88rem;color:#f8fafc}
[data-testid="stForm"]{border:1px solid #35435e;border-radius:18px;background:rgba(255,255,255,.025);padding:20px}
header[data-testid="stHeader"]{background:transparent}
@media(max-width:700px){.winner-panel{padding:20px}.score{font-size:2.7rem}.winner-panel h2{font-size:1.6rem}.score span{display:block}}
 </style>""", unsafe_allow_html=True)
st.markdown('<div class="brand"><h1>WineLab</h1><p>Geometría, márgenes y redes para explorar la calidad del vino.</p></div>', unsafe_allow_html=True)
SECTIONS = ["Mejor modelo", "Comparativa de modelos", "Simulador modular", "EDA y PCA", "Recorrido técnico", "Descargar predicciones"]
STEPS = ["1 · Resumen del proyecto", "2 · EDA y limpieza", "3 · Baseline por distancias",
         "4 · Variables nuevas y PCA", "5 · SVM y MLP", "6 · Resultados y predicciones"]
METHODS = ["Elegir en el simulador", "k-NN", "k-means", "PCA", "SVM lineal", "SVM polinomial", "SVM RBF", "MLP"]


def route_method():
    st.session_state['section'] = "Simulador modular"
    st.session_state.pop('selected_model', None)


with st.sidebar:
    st.subheader("Explora WineLab")
    section = st.radio("Sección", SECTIONS, key="section")
    st.divider()
    method = st.selectbox("¿Qué método quieres usar?", METHODS, key="method", on_change=route_method)
    if section == "Recorrido técnico":
        step = st.radio("Módulo técnico", STEPS, key="step")
    st.caption("Modelos guardados: no hace falta entrenar para predecir.")
    st.caption("Clasificación binaria · semilla 42 · F1 macro")
try:
    train, test = load_data()
    results, meta, specs = saved_experiment()
    data = consolidated()
    if section == "Mejor modelo":
        winner_panel(data)
    elif section == "Comparativa de modelos":
        comparison(data)
    elif section == "Simulador modular":
        modular_simulation(data, train, meta, results, specs, method)
    elif section == "EDA y PCA":
        st.header("EDA y PCA")
        render_pca_overview(train)
        st.subheader("Distribución de clases")
        a,b = st.columns(2)
        with a:
            show_figure(class_figure(train))
        with b:
            st.metric("Observaciones de entrenamiento", str(len(train)))
            st.metric("Mediciones originales", "11")
            st.write("F1 macro asigna el mismo peso a ambas clases. El id no es un predictor.")
    elif section == "Recorrido técnico":
        pages = [setup, eda, baseline, features, advanced, delivery]
        st.caption("Progresión matemática del proyecto base. La selección entre las tres versiones aparece en «Mejor modelo» y en el módulo 6.")
        pages[STEPS.index(step)].render(train, test, results, meta, specs, method)
    else:
        st.header("Predicciones del mejor modelo")
        winner_panel(data, compact=True)
        frame = pd.read_csv(ROOT / "entregas/submission.csv")
        validate_submission(frame, test)
        st.success("1.300 predicciones · id,prediction · sin nulos · orden original conservado")
        st.download_button("Descargar submission.csv", frame.to_csv(index=False), "submission.csv", "text/csv", type="primary")
except (ValueError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.info("Comprueba data/train_public.csv y data/test_public_features.csv. Luego ejecuta python scripts/entrenar.py.")
