"""Evaluación sobre folds comunes y generación verificable de la entrega."""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from src.data import FEATURES, ROOT, SEED, audit_data, load_data, validate_submission
from src.models import build_pipeline, candidates


def json_safe(value):
    """JSON estándar: los valores faltantes de pandas se exportan como null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def evaluate_candidate(spec, X, y, splits):
    rows, out_of_fold = [], np.full(len(y), -1, dtype=int)
    for fold, (train_index, valid_index) in enumerate(splits, 1):
        pipeline = build_pipeline(spec)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            start = perf_counter()
            pipeline.fit(X.iloc[train_index], y.iloc[train_index])
            fit_seconds = perf_counter() - start
            start = perf_counter()
            pred = pipeline.predict(X.iloc[valid_index])
            prediction_seconds = perf_counter() - start
        out_of_fold[valid_index] = pred
        precision, recall, f1, _ = precision_recall_fscore_support(
            y.iloc[valid_index], pred, labels=[0, 1], zero_division=0)
        rows.append({
            "name": spec.name, "fold": fold, "stage": spec.stage, "family": spec.family,
            "f1_macro": float(np.mean(f1)), "f1_0": f1[0], "f1_1": f1[1],
            "precision_1": precision[1], "recall_1": recall[1],
            "accuracy": accuracy_score(y.iloc[valid_index], pred),
            "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
            "input_dimensions": pipeline[:-1].transform(X.iloc[valid_index[:1]]).shape[1],
            "warnings_count": len(caught),
            "warnings": " | ".join(sorted({str(item.message) for item in caught})),
        })
    if (out_of_fold < 0).any():
        raise ValueError("Cada observación debe tener exactamente una predicción fuera de fold.")
    detail = pd.DataFrame(rows)
    summary = {"name": spec.name, "stage": spec.stage, "family": spec.family,
               "ratios": spec.ratios, "pca_variance": spec.pca, "centroids": spec.centroids,
               "params": json.dumps(spec.params, sort_keys=True),
               "f1_macro_mean": detail.f1_macro.mean(),
               "f1_macro_std": detail.f1_macro.std(ddof=1),
               "f1_0_mean": detail.f1_0.mean(), "f1_1_mean": detail.f1_1.mean(),
               "precision_1_mean": detail.precision_1.mean(), "recall_1_mean": detail.recall_1.mean(),
               "accuracy_mean": detail.accuracy.mean(),
               "fit_seconds_mean": detail.fit_seconds.mean(),
               "prediction_seconds_mean": detail.prediction_seconds.mean(),
               "dimensions_mean": detail.input_dimensions.mean(),
               "warnings_count": int(detail.warnings_count.sum())}
    return summary, detail, out_of_fold


def run_experiment(profile="complete", folds=5, output_dir=None):
    out = Path(output_dir) if output_dir else ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    train, test = load_data()
    X, y = train[FEATURES], train.target.astype(int)
    specs = candidates(profile)
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED).split(X, y))
    fold_assignment = np.zeros(len(train), dtype=int)
    for fold, (_, index) in enumerate(splits, 1):
        fold_assignment[index] = fold
    pd.DataFrame({"id": train.id, "fold": fold_assignment}).to_csv(out / "fold_assignments.csv", index=False)
    audit = audit_data(train, test)
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "candidates.json").write_text(json.dumps([s.to_dict() for s in specs], indent=2), encoding="utf-8")
    all_summaries, all_detail, predictions = [], [], {}
    started = perf_counter()
    for index, spec in enumerate(specs, 1):
        summary, detail, oof = evaluate_candidate(spec, X, y, splits)
        all_summaries.append(summary)
        all_detail.append(detail)
        predictions[spec.name] = oof
        # Checkpoint legible por la app incluso durante el entrenamiento.
        pd.DataFrame(all_summaries).to_csv(out / "cv_results.csv", index=False)
        print(f"[{index:02}/{len(specs)}] {spec.name}: F1 macro={summary['f1_macro_mean']:.4f} "
              f"(+/- {summary['f1_macro_std']:.4f}), ajuste={summary['fit_seconds_mean']:.2f}s", flush=True)
    results = pd.DataFrame(all_summaries).sort_values(
        ["f1_macro_mean", "f1_macro_std", "name"], ascending=[False, True, True]).reset_index(drop=True)
    results.to_csv(out / "cv_results.csv", index=False)
    pd.concat(all_detail, ignore_index=True).to_csv(out / "cv_folds.csv", index=False)
    lookup = {s.name: s for s in specs}
    best = lookup[results.iloc[0]["name"]]
    pd.DataFrame({"id": train.id, "target": y, "prediction": predictions[best.name],
                  "fold": fold_assignment}).to_csv(out / "best_oof_predictions.csv", index=False)
    # Control adicional: perfiles idénticos no pueden caer en train y validación.
    # La guía exige StratifiedKFold: este control no cambia la selección oficial.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    group_splits = list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED).split(X, y, groups))
    grouped, grouped_detail, _ = evaluate_candidate(best, X, y, group_splits)
    grouped_detail.to_csv(out / "grouped_validation_folds.csv", index=False)
    # Guardar mejor por familia y por kernel para que el selector funcione de inmediato.
    chosen = {best.name}
    for family in results.family.unique():
        chosen.add(results[results.family == family].iloc[0]["name"])
    for kernel in ("linear", "poly", "rbf"):
        names = [s.name for s in specs if s.family == "SVM" and s.params["kernel"] == kernel]
        chosen.add(results[results.name.isin(names)].iloc[0]["name"])
    artifacts = {}
    for name in sorted(chosen):
        model = build_pipeline(lookup[name])
        with warnings.catch_warnings(record=True) as final_warnings, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            model.fit(X, y)
        artifact = {"pipeline": model, "spec": lookup[name].to_dict(), "sha256": audit["sha256"],
                    "sklearn_version": sklearn.__version__,
                    "fit_warnings": [str(w.message) for w in final_warnings]}
        filename = f"model_{name}.joblib"
        joblib.dump(artifact, out / filename)
        artifacts[name] = filename
    artifact = joblib.load(out / artifacts[best.name])
    with threadpool_limits(limits=1):
        final_predictions = artifact["pipeline"].predict(test[FEATURES]).astype(int)
    submission = pd.DataFrame({"id": test.id, "prediction": final_predictions})
    validate_submission(submission, test)
    # Los perfiles de diagnóstico se mantienen separados de la entrega definitiva.
    delivery = ROOT / "entregas" if output_dir is None else out / "entregas"
    delivery.mkdir(parents=True, exist_ok=True)
    submission.to_csv(delivery / "submission.csv", index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "profile": profile,
        "seed": SEED, "folds": folds, "candidates": len(specs),
        "splitter": "StratifiedKFold(shuffle=True, random_state=42)",
        "selection_rule": "Mayor F1 macro medio; empate: menor desviación, luego nombre.",
        "best": best.to_dict(), "best_cv": results.iloc[0].to_dict(),
        "grouped_diagnostic": grouped,
        "grouped_note": "Diagnóstico del ganador ya seleccionado; no es evaluación anidada ni test independiente.",
        "oof_confusion_matrix": confusion_matrix(y, predictions[best.name], labels=[0, 1]).tolist(),
        "artifacts": artifacts, "sha256": audit["sha256"],
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "elapsed_seconds": perf_counter() - started,
        "hidden_test_f1_macro": None,
    }
    metadata = json_safe(metadata)
    (out / "experiment.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Ganador: {best.name}. Entrega validada: {len(submission)} observaciones.", flush=True)
    return results, metadata
