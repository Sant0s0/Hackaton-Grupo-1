"""El material de clase debe cargar modelos preparados, sin ajustar en vivo."""
import json
import numpy as np
import pytest
from src.data import ROOT, FEATURES, fingerprints, load_data
from src.models import Candidate
from src.persistence import load_saved, model_path, prepare_model


def test_every_evaluated_configuration_loads_without_training(monkeypatch):
    train, test = load_data()
    hashes = fingerprints()
    specs = [Candidate(**item) for item in json.loads((ROOT / "outputs/candidates.json").read_text())]
    manifest = json.loads((ROOT / "outputs/demo_models/manifest.json").read_text())
    assert set(manifest["models"]) == {s.name for s in specs}
    assert manifest["sha256"] == hashes
    def forbidden_fit(*args, **kwargs):
        pytest.fail("Una configuración preparada intentó volver a entrenar.")
    monkeypatch.setattr("src.persistence.build_pipeline", forbidden_fit)
    for spec in specs:
        artifact = prepare_model(spec, train, hashes)
        prediction = artifact["pipeline"].predict(test[FEATURES].iloc[:3])
        assert np.isin(prediction, [0, 1]).all()


def test_custom_model_survives_a_fresh_load(tmp_path, monkeypatch):
    train, _ = load_data()
    hashes = fingerprints()
    spec = Candidate("custom_example", 3, "k-NN", {"n_neighbors": 3})
    prepared = prepare_model(spec, train.iloc[:100], hashes, directory=tmp_path)
    expected = prepared["pipeline"].predict(train[FEATURES].iloc[100:105])
    monkeypatch.setattr("src.persistence.build_pipeline", lambda *args: pytest.fail("Reajuste inesperado"))
    loaded = prepare_model(spec, train.iloc[:100], hashes, directory=tmp_path)
    np.testing.assert_array_equal(loaded["pipeline"].predict(train[FEATURES].iloc[100:105]), expected)
    assert model_path(spec, hashes["train_public.csv"], tmp_path).is_file()
    assert load_saved(spec, {"train_public.csv": "other_data"}, directory=tmp_path) is None


def test_saved_base_winner_matches_archived_submission():
    import pandas as pd
    _, test = load_data()
    meta = json.loads((ROOT / "outputs/experiment.json").read_text(encoding="utf-8"))
    artifact = load_saved(Candidate(**meta["best"]), fingerprints())
    actual = artifact["pipeline"].predict(test[FEATURES])
    expected = pd.read_csv(ROOT / "outputs/submission_base.csv").prediction
    np.testing.assert_array_equal(actual, expected)
