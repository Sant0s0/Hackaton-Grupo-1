import streamlit as st


def render(train, test, results, meta, specs, method):
    st.header("1 · Resumen del proyecto")
    st.write("Clasificar vinos de calidad estándar (0) o superior (1) a partir de once mediciones fisicoquímicas.")
    cols = st.columns(4)
    cols[0].metric("Vinos para aprender", f"{len(train):,}")
    cols[1].metric("Vinos por predecir", f"{len(test):,}")
    cols[2].metric("Variables originales", "11")
    cols[3].metric("Métrica oficial", "F1 macro")
    st.subheader("Del laboratorio a la predicción")
    st.write("Once propiedades de cada vino se transforman en una representación numérica. Comparamos métodos de distancia, margen y redes neuronales para distinguir las dos clases.")
    st.table({"Método": ["k-NN", "k-means", "PCA", "SVM", "MLP"],
              "Qué puedes explorar": ["Vecinos, distancias y votación", "Centroides y clase mayoritaria",
                                      "Proyección y varianza retenida", "Kernels lineal, polinomial y RBF", "Capas y activaciones"]})
    if meta:
        st.info(f"Hay {meta['candidates']} configuraciones evaluadas en {meta['folds']} folds. El test oculto no tiene etiquetas disponibles.")
