import pandas as pd
import streamlit as st
from app.common import metrics_table, model_panel, show_figure
from src.data import ROOT, validate_submission
from src.models import Candidate
from src.plots import confusion_figure


def render(train, test, results, meta, specs, method):
    st.header("6 · Comparación, entrega y pitch")
    if results.empty or not meta:
        st.info("Ejecuta python scripts/entrenar.py para generar la entrega.")
        return
    st.subheader("Comparación completa · mismos cinco folds")
    metrics_table(results)
    st.download_button("Descargar resultados de validación", results.to_csv(index=False), "cv_results.csv", "text/csv")
    st.write(f"Modelo elegido: **{meta['best']['name']}**. Se vuelve a ajustar sobre las 5.197 filas de entrenamiento para predecir el test.")
    left, right = st.columns(2)
    with left:
        show_figure(confusion_figure(meta["oof_confusion_matrix"]))
    with right:
        st.metric("F1 macro · selección por CV", f"{meta['best_cv']['f1_macro_mean']:.4f}")
        st.metric("F1 macro · control por perfiles", f"{meta['grouped_diagnostic']['f1_macro_mean']:.4f}")
        st.caption("El segundo valor agrupa perfiles idénticos. Es un diagnóstico del modelo elegido, no un test independiente. La selección entre muchas configuraciones puede producir optimismo.")
        st.info("El F1 macro del test oculto solo lo puede calcular la profesora con sus etiquetas.")
    submission = pd.read_csv(ROOT / "entregas/submission.csv")
    validate_submission(submission, test)
    st.success(f"CSV comprobado: {len(submission):,} vinos, columnas id/prediction, sin nulos y en el orden original.")
    st.download_button("Descargar submission.csv", submission.to_csv(index=False), "submission.csv", "text/csv", type="primary")
    st.subheader("Simulador del modelo entregado")
    model_panel([Candidate(**meta["best"])], results, train, meta, "winner")
    st.subheader("Material para entender y defender el trabajo")
    for path, title in [(ROOT / "docs/EXPLICACION.html", "Documento explicado paso a paso"),
                        (ROOT / "docs/PITCH.html", "Guion del pitch de cinco minutos")]:
        if path.exists():
            st.download_button(title, path.read_bytes(), path.name, "text/html")
