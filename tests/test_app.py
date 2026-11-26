"""Ejercita los seis módulos y predictores con el motor real de Streamlit."""
import pytest
from streamlit.testing.v1 import AppTest
from src.data import ROOT

pytestmark = pytest.mark.skipif(not (ROOT / "outputs/experiment.json").exists(), reason="Entrenar primero")


def test_all_six_modules_open():
    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()
    assert not at.exception
    for section in at.radio(key="section").options:
        at.radio(key="section").set_value(section).run()
        assert not at.exception
    at.radio(key="section").set_value("Recorrido técnico").run()
    options = list(at.radio(key="step").options)
    for option in options:
        at.radio(key="step").set_value(option).run()
        assert not at.exception, f"Falló {option}: {at.exception}"
        assert at.header[0].value.startswith(option.split(" · ")[0])


@pytest.mark.parametrize("method", ["k-NN", "k-means", "PCA", "SVM lineal", "SVM polinomial", "SVM RBF", "MLP"])
def test_method_selector_and_prediction(method):
    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=45).run()
    at.selectbox(key="method").set_value(method).run()
    assert not at.exception
    predict_button = next(button for button in at.button if button.label == "Predecir calidad")
    predict_button.click().run()
    assert not at.exception
    assert any("clase" in success.value for success in at.success)


def test_custom_parameters_and_navigation_reset():
    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()
    at.selectbox(key="method").set_value("k-NN").run()
    at.toggle[0].set_value(True).run()
    metric = next(box for box in at.selectbox if box.label == "Distancia / disimilitud")
    metric.set_value("minkowski").run()
    assert not at.exception
    assert any(slider.label == "Orden p de Minkowski" for slider in at.slider)
    at.radio(key="section").set_value("EDA y PCA").run()
    assert not at.exception
    assert at.radio(key="section").value == "EDA y PCA"


def test_app_keeps_group_material_out_of_the_demonstration():
    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()
    for section in ("Mejor modelo", "Descargar predicciones"):
        at.radio(key="section").set_value(section).run()
        assert not at.exception
        visible = " ".join(item.value for collection in [at.markdown, at.header, at.subheader, at.caption]
                           for item in collection)
        assert all(word not in visible.lower() for word in ["guion", "pitch", "responsabilidades", "docs/grupo"])


def test_complete_ranking_is_visible_and_includes_all_configurations():
    import json
    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=45).run()
    at.radio(key="section").set_value("Comparativa de modelos").run()
    assert not at.exception
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert len(spec['data'][0]['x']) == 116
    assert spec['layout']['yaxis']['autorange'] == 'reversed'
    table = at.dataframe[0].value
    assert len(table) == 116
    assert list(table.columns) == ['Puesto', 'Configuración', 'F1 macro']
    assert table.iloc[0]['F1 macro'] == table['F1 macro'].max()
    assert any(h.value == 'Ranking completo de modelos' for h in at.subheader)


def test_pca_spectrum_and_thresholds_for_both_representations():
    import json
    import numpy as np
    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=45).run()
    at.radio(key="section").set_value("EDA y PCA").run()
    assert not at.exception
    fig = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert [trace['type'] for trace in fig['data']] == ['bar', 'scatter']
    assert {shape['y0'] for shape in fig['layout']['shapes']} == {90,95}
    table = at.dataframe[0].value
    assert len(table) == 11
    assert next(m.value for m in at.metric if m.label == 'Para conservar el 90%') == '7 componentes'
    assert next(m.value for m in at.metric if m.label == 'Para conservar el 95%') == '9 componentes'
    assert np.isclose(table['Varianza individual (%)'].sum(),100)
    assert np.isclose(table['Varianza acumulada (%)'].iloc[-1],100)
    at.radio(key='pca_representation_main').set_value('22 atributos químicos de Alejo').run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 22
