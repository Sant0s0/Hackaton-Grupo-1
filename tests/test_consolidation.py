import json
import joblib
import numpy as np
import pandas as pd
from src.data import ROOT, FEATURES, load_data, fingerprints, validate_submission


def test_final_selection_and_submission_without_fit(monkeypatch):
    from sklearn.neighbors import KNeighborsClassifier
    from src.pipeline_wine import PhysicochemicalFeatures, KMeansCentroidDistances
    def forbidden(*args, **kwargs):
        raise AssertionError("No debe entrenar durante la presentación")
    for cls in [KNeighborsClassifier, PhysicochemicalFeatures, KMeansCentroidDistances]:
        monkeypatch.setattr(cls, "fit", forbidden)
    train, test = load_data()
    meta = json.loads((ROOT / "outputs/consolidated.json").read_text(encoding="utf-8"))
    assert meta['sha256'] == fingerprints()
    assert meta['total_configurations'] == 116
    comparison = pd.read_csv(ROOT / "outputs/project_comparison.csv")
    winner = next(m for m in meta['models'] if m['id'] == meta['winner_id'])
    assert np.isclose(winner['f1_macro'], comparison.f1_macro.max())
    assert winner['params']['n_neighbors'] == 23
    for item in meta['models']:
        pipe = joblib.load(ROOT / "outputs" / item['artifact'])['pipeline']
        assert np.isin(pipe.predict(test[FEATURES].iloc[:4]), [0, 1]).all()
    pipe = joblib.load(ROOT / "outputs" / winner['artifact'])['pipeline']
    assert pipe[:-1].transform(train[FEATURES].iloc[:1]).shape[1] == 40
    actual = pd.read_csv(ROOT / "entregas/submission_Grupo_1.csv")
    validate_submission(actual, test)
    np.testing.assert_array_equal(pipe.predict(test[FEATURES]), actual.prediction)
    assert meta['import_verification']['source_submission_reproduced']
