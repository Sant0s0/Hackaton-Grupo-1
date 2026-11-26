"""Punto de entrada: streamlit run app/app.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from app.common import saved_experiment
from app.modules import setup, eda, baseline, features, advanced, delivery
from src.data import load_data

st.set_page_config(page_title="VinoLab · Métodos matemáticos para IA", page_icon="🍇", layout="wide")
st.markdown("""<style>
 .block-container{max-width:1200px;padding-top:2.2rem}
 h1,h2,h3{letter-spacing:-.025em} h1{color:#293949}
 [data-testid="stMetric"]{background:#eef3f7;border:1px solid #dce5ed;border-radius:12px;padding:16px}
 [data-testid="stSidebar"]{border-right:1px solid #dce5ed}
 .stCaption{line-height:1.5} [data-testid="stForm"]{border-radius:14px}
 </style>""", unsafe_allow_html=True)
st.title("VinoLab")
st.caption("HACKATHON · MÉTODOS Y MODELOS MATEMÁTICOS PARA IA · JAVERIANA CALI")
STEPS = ["1 · Configuración y roles", "2 · EDA y limpieza", "3 · Baseline por distancias",
         "4 · Variables nuevas y PCA", "5 · SVM y MLP", "6 · Entrega y pitch"]
METHODS = ["Seguir el paso seleccionado", "k-NN", "k-means", "PCA", "SVM lineal", "SVM polinomial", "SVM RBF", "MLP"]


def reset_method():
    st.session_state["method"] = METHODS[0]


def route_method():
    method = st.session_state["method"]
    if method in ("k-NN", "k-means"):
        st.session_state["step"] = STEPS[2]
    elif method == "PCA":
        st.session_state["step"] = STEPS[3]
    elif method.startswith("SVM") or method == "MLP":
        st.session_state["step"] = STEPS[4]


with st.sidebar:
    st.subheader("Explora el trabajo")
    st.radio("Paso de la guía", STEPS, key="step", on_change=reset_method)
    st.divider()
    method = st.selectbox("¿Qué método quieres usar?", METHODS, key="method", on_change=route_method)
    st.caption("Orden oficial de la guía. Cada módulo permite comparar y probar sus métodos.")
    st.caption("Clasificación binaria · semilla 42 · F1 macro")
try:
    train, test = load_data()
    results, meta, specs = saved_experiment()
    pages = [setup, eda, baseline, features, advanced, delivery]
    pages[STEPS.index(st.session_state["step"])].render(train, test, results, meta, specs, method)
except (ValueError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.info("Comprueba data/train_public.csv y data/test_public_features.csv. Luego ejecuta python scripts/entrenar.py.")
