"""Controles sobre fuga, contratos y correspondencia clúster/clase."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from src.data import FEATURES, ROOT, load_data, validate_frame, validate_submission, fingerprints
from src.features import ChemicalRatios
from src.models import Candidate, KMeansMajority, build_pipeline, candidates


def test_scaler_and_imputer_learn_only_training_rows():
    train, _ = load_data()
    x = train[FEATURES].iloc[:100].copy()
    held_out = train[FEATURES].iloc[100:110].copy() * 1000
    x.loc[x.index[0], "alcohol"] = np.nan
    pipe = build_pipeline(Candidate("test", 3, "k-NN", {"n_neighbors": 3}))
    pipe.fit(x, train.target.iloc[:100])
    expected = x.fillna(x.median()).mean().to_numpy()
    np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, expected)
    old_mean = pipe.named_steps["scaler"].mean_.copy()
    pipe.predict(held_out)
    np.testing.assert_array_equal(pipe.named_steps["scaler"].mean_, old_mean)


def test_zero_denominators_are_imputed_and_not_infinite():
    train, _ = load_data()
    x = train[FEATURES].iloc[:20].copy()
    x.loc[x.index[0], ["volatile_acidity", "total_sulfur_dioxide", "density"]] = 0
    expanded = ChemicalRatios().fit_transform(x)
    assert expanded.shape == (20, 14)
    assert np.isnan(expanded[0, -3:]).all()
    assert not np.isinf(expanded).any()
    pipe = build_pipeline(Candidate("test", 4, "k-NN", {"n_neighbors": 3}, ratios=True))
    pipe.fit(x, train.target.iloc[:20])
    assert np.isfinite(pipe[:-1].transform(x)).all()


def test_clusters_are_mapped_to_training_class_not_cluster_id():
    x = np.array([[-10.], [-9.], [-8.], [8.], [9.], [10.]])
    y = np.array([1, 1, 1, 0, 0, 0])
    model = KMeansMajority(n_clusters=2).fit(x, y)
    np.testing.assert_array_equal(model.predict(x), y)
    np.testing.assert_array_equal(clone(model).fit(x, 1-y).predict(x), 1-y)
    np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1)


def test_submission_rejects_missing_reordered_or_extra_rows():
    _, test = load_data()
    good = pd.DataFrame({"id": test.id, "prediction": np.zeros(len(test), dtype=int)})
    assert validate_submission(good, test)
    for wrong in [good.iloc[::-1].reset_index(drop=True), good.iloc[:-1],
                  good.assign(prediction=2), good.assign(target=0)]:
        with pytest.raises(ValueError):
            validate_submission(wrong, test)


def test_data_contract_rejects_id_as_feature_and_target_in_test():
    train, test = load_data()
    assert "id" not in FEATURES and "target" not in FEATURES
    with pytest.raises(ValueError):
        validate_frame(test.assign(target=0))
    with pytest.raises(ValueError):
        validate_frame(train.assign(id="duplicado"), training=True)


def test_registry_covers_required_methods_and_baseline_grid():
    specs = candidates()
    assert len({s.name for s in specs}) == len(specs)
    assert {s.family for s in specs} == {"k-NN", "k-means", "SVM", "MLP"}
    assert {s.params["metric"] for s in specs if s.family == "k-NN"} == {
        "euclidean", "manhattan", "minkowski", "cosine"}
    assert {(s.params["n_neighbors"], s.params["metric"]) for s in specs if s.stage == 3 and s.family == "k-NN"} == {
        (k, metric) for k in (3, 5, 7) for metric in ("euclidean", "manhattan")}
    for spec in specs:
        clone(build_pipeline(spec))


@pytest.mark.skipif(not (ROOT / "outputs/experiment.json").exists(), reason="Requiere entrenamiento completo")
def test_saved_artifacts_reproduce_the_submission():
    _, test = load_data()
    meta = json.loads((ROOT / "outputs/experiment.json").read_text(encoding="utf-8"))
    assert meta["sha256"] == fingerprints()
    artifact = joblib.load(ROOT / "outputs" / meta["artifacts"][meta["best"]["name"]])
    predictions = artifact["pipeline"].predict(test[FEATURES])
    actual = pd.read_csv(ROOT / "outputs/submission_base.csv")
    assert validate_submission(actual, test)
    np.testing.assert_array_equal(predictions, actual.prediction)
    folds = pd.read_csv(ROOT / "outputs/cv_folds.csv")
    assert len(folds) == meta["candidates"] * meta["folds"]
    assert folds.groupby("name").fold.nunique().eq(meta["folds"]).all()
    means = folds.groupby("name").f1_macro.mean()
    assert means.idxmax() == meta["best"]["name"]
