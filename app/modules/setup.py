import streamlit as st


def render(train, test, results, meta, specs, method):
    st.header("1 · Configuración y roles")
    st.write("Clasificar vinos de calidad estándar (0) o superior (1) a partir de once mediciones fisicoquímicas.")
    cols = st.columns(4)
    cols[0].metric("Vinos para aprender", f"{len(train):,}")
    cols[1].metric("Vinos por predecir", f"{len(test):,}")
    cols[2].metric("Variables originales", "11")
    cols[3].metric("Métrica oficial", "F1 macro")
    st.subheader("El recorrido de la guía")
    st.write("Usa los pasos de la barra lateral para recorrer el trabajo en orden. Para explorar un método directamente, abre el selector «¿Qué método quieres usar?». Cada método tiene su explicación, configuraciones y simulador.")
    st.table({"Paso": ["1. Configuración", "2. EDA y limpieza", "3. Baseline", "4. Variables y PCA", "5. SVM y MLP", "6. Entrega y pitch"],
              "Tiempo de la guía": ["0–10 min", "10–30 min", "30–50 min", "50–75 min", "75–100 min", "100–120 min"]})
    st.subheader("Distribución de responsabilidades sugerida")
    st.table({"Responsabilidad": ["Datos y matemática", "Modelos y validación", "Aplicación y defensa"],
              "Qué debe verificar": ["EDA, unidades, ratios, ausencia de fuga", "Baseline, folds comunes, SVM/MLP y F1 macro", "Streamlit, CSV, README y pitch"]})
    st.caption("Roles propuestos, sin asignar nombres ni atribuir contribuciones que no se han registrado.")
    if meta:
        st.info(f"Hay {meta['candidates']} configuraciones evaluadas en {meta['folds']} folds. El test oculto no tiene etiquetas disponibles.")
