"""Ejercita los seis módulos y predictores con el motor real de Streamlit."""
import pytest
from streamlit.testing.v1 import AppTest
from src.data import ROOT

pytestmark = pytest.mark.skipif(not (ROOT / "outputs/experiment.json").exists(), reason="Entrenar primero")


def test_all_six_modules_open():
    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()
    assert not at.exception
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
    at.radio(key="step").set_value(at.radio(key="step").options[1]).run()
    assert not at.exception
    assert at.selectbox(key="method").value == "Seguir el paso seleccionado"
