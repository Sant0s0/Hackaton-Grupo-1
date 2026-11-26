import pandas as pd
import streamlit as st
from app.common import metrics_table, model_panel, show_figure
from src.data import ROOT, validate_submission
from src.models import Candidate
from src.plots import confusion_figure


def render(train, test, results, meta, specs, method):
    st.header("6 · Resultados y predicciones")
    from app.dashboard import consolidated, winner_panel, comparison
    data = consolidated()
    winner_panel(data, compact=True)
    comparison(data)
    submission = pd.read_csv(ROOT / "entregas/submission.csv")
    validate_submission(submission, test)
    st.success(f"CSV comprobado: {len(submission):,} vinos, columnas id/prediction, sin nulos y en el orden original.")
    st.download_button("Descargar submission.csv", submission.to_csv(index=False), "submission.csv", "text/csv", type="primary")
