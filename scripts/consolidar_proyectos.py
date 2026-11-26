"""Integra resultados existentes y pipelines guardados, sin búsqueda ni entrenamiento."""
from pathlib import Path
import ast
import base64
import gzip
import hashlib
import io
import json
import shutil
import sys
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data import ROOT, FEATURES, fingerprints, load_data, validate_submission


def main():
    alejo = ROOT / "hackaton_alejoversion/hackaton"
    other = ROOT / "Hackathon_MMMIA_Grupo1 (1)"
    train, test = load_data()
    # Verificación por valores, columnas y orden: los CSV pueden tener distintos saltos de línea.
    for name, expected in [("train_public.csv", train), ("test_public_features.csv", test)]:
        pd.testing.assert_frame_equal(pd.read_csv(alejo / name), expected)
    notebook = json.loads((other / "notebooks/01_eda_y_modelado.ipynb").read_text(encoding="utf-8"))
    for name, expected in [("train_public.csv", train), ("test_public_features.csv", test)]:
        packed = notebook["metadata"]["datos_csv_gzip"][name]
        pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(packed)))), expected)
    second = next(o["data"]["application/vnd.mmmia.resultados+json"]
                  for c in notebook["cells"] for o in c.get("outputs", [])
                  if "application/vnd.mmmia.resultados+json" in o.get("data", {}))
    summary = json.loads((alejo / "models/benchmark_summary.json").read_text(encoding="utf-8"))
    original = pd.read_csv(ROOT / "outputs/cv_results.csv")
    rows = []
    for _, r in original.iterrows():
        rows.append({"source": "Proyecto base", "id": r['name'], "name": r['name'], "family": r.family,
                     "f1_macro": r.f1_macro_mean, "std": r.f1_macro_std,
                     "reported_f1": r.f1_macro_mean, "adjusted_threshold": False,
                     "fit_seconds": r.fit_seconds_mean})
    for r in second["metricas"]:
        family = "k-NN" if r["familia"].startswith("k-NN") else r["familia"]
        rows.append({"source": "Versión alternativa", "id": r["modelo"], "name": r["modelo"],
                     "family": family, "f1_macro": r["f1_macro"], "std": r["sd_f1"],
                     "reported_f1": r["f1_macro"], "adjusted_threshold": False, "fit_seconds": r["ajuste_s"]})
    for r in summary["models_ranked"]:
        family = "k-NN" if "knn" in r["id"] else ("MLP" if r["id"].startswith("mlp") else
                   "k-means" if r["id"].startswith("kmeans") else "SVM")
        # La fuente usa ddof=0; normalizamos a desviación muestral si corresponde al score estándar.
        sd = r["f1_macro_std"] * np.sqrt(5 / 4) if not r["usa_umbral_calibrado"] else None
        rows.append({"source": "Versión Alejo", "id": r["id"], "name": r["nombre"], "family": family,
                     "f1_macro": r["f1_macro_default"], "std": sd, "reported_f1": r["f1_macro_cv"],
                     "adjusted_threshold": r["usa_umbral_calibrado"], "fit_seconds": r["fit_time_mean_s"]})
    table = pd.DataFrame(rows).sort_values(["f1_macro", "source", "id"], ascending=[False, True, True]).reset_index(drop=True)
    table.to_csv(ROOT / "outputs/project_comparison.csv", index=False)
    # Solo definiciones necesarias para deserializar artefactos propios aportados por el equipo.
    # No se copia ni ejecuta el entrenamiento o la calibración de la otra app.
    text = (alejo / "src/pipeline_wine.py").read_text(encoding="utf-8")
    wanted = {"PhysicochemicalFeatures", "KMeansCentroidDistances", "KMeansClusterClassifier",
              "CalibratedThresholdModel", "extract_continuous_score"}
    definitions = [ast.get_source_segment(text, node) for node in ast.parse(text).body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    definitions = [definition.replace("sin fuga de datos mediante predicciones Out-of-Fold en train.",
                    "en la versión de origen. La consolidación usa el pipeline sin ajustar umbral.") for definition in definitions]
    preamble = '''"""Transformaciones reutilizadas de la versión Alejo; sin ejecutar su entrenamiento.

CalibratedThresholdModel se conserva solo para leer sus archivos originales.
La consolidación extrae el pipeline aprendido y usa predict con decisión estándar.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from src.data import FEATURES as RAW_FEATURES

FEATURE_METADATA = {name: {"default": 1.0} for name in RAW_FEATURES}

'''
    (ROOT / "src/pipeline_wine.py").write_text(preamble + "\n\n".join(definitions) + "\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        winner_wrapper = joblib.load(alejo / "models/best_pipeline.joblib")
        reps = joblib.load(alejo / "models/representative_models.joblib")
    # Chequeo de compatibilidad del archivo importado contra la entrega de su propia fuente.
    with threadpool_limits(limits=1):
        source_predictions = winner_wrapper.predict(test[FEATURES])
    source_submission = pd.read_csv(alejo / "entregas/submission.csv")
    validate_submission(source_submission, test)
    np.testing.assert_array_equal(source_predictions, source_submission.prediction)
    meta = json.loads((ROOT / "outputs/experiment.json").read_text(encoding="utf-8"))
    dest = ROOT / "outputs/selected_models"
    dest.mkdir(parents=True, exist_ok=True)
    entries = []

    def add_saved(identifier, name, family, origin, pipeline, score, params, preprocessing, std=None):
        filename = f"{identifier}.joblib"
        artifact = {"pipeline": pipeline, "params": params, "sha256": fingerprints(), "name": name}
        joblib.dump(artifact, dest / filename, compress=3)
        entries.append({"id": identifier, "name": name, "family": family, "source": origin,
                        "artifact": "selected_models/" + filename, "f1_macro": float(score),
                        "std": None if std is None else float(std), "params": params,
                        "preprocessing": preprocessing})

    aw = summary["models_ranked"][0]
    assert aw["id"] == "fe_kmeans_knn_cos" and not aw["usa_umbral_calibrado"]
    add_saved("selected_knn", "k-NN · coseno + centroides", "k-NN", "Versión Alejo", winner_wrapper.pipeline_,
              aw["f1_macro_default"], {"n_neighbors": 23, "metric": "cosine", "weights": "distance",
              "centroid_groups": [2, 6], "use_log": True, "PCA": False},
              "11 mediciones → 22 variables (6 derivadas y 5 logaritmos) → z-score → distancias a centroides 2/6 y nuevo z-score → k-NN",
              aw["f1_macro_std"] * np.sqrt(5/4))
    for identifier, kernel in [("svm_rbf_spherical", "rbf"), ("svm_poly_fe", "poly")]:
        model = reps[identifier]["model"].pipeline_
        r = next(item for item in summary["models_ranked"] if item["id"] == identifier)
        params = {k: v for k, v in model.named_steps["clf"].get_params().items()
                  if k in ["C", "gamma", "kernel", "degree", "coef0", "class_weight"]}
        params["threshold"] = 0.0
        add_saved(f"selected_svm_{kernel}", f"SVM · {kernel.upper()}", "SVM", "Versión Alejo", model,
                  r["f1_macro_default"], params,
                  "22 variables químicas/logarítmicas → z-score" + (" → normalización L2" if kernel == "rbf" else "") + " → SVM; umbral estándar")
    for family, identifier in [("MLP", "selected_mlp"), ("k-means", "selected_kmeans")]:
        row = original[original.family == family].sort_values("f1_macro_mean", ascending=False).iloc[0]
        artifact = joblib.load(ROOT / "outputs" / meta["artifacts"][row['name']])
        spec = artifact['spec']
        add_saved(identifier, "MLP · ReLU (64, 32)" if family == "MLP" else "k-means · mayoría", family,
                  "Proyecto base", artifact["pipeline"], row.f1_macro_mean, spec['params'],
                  "11 mediciones" + (" + 3 cocientes" if spec['ratios'] else "") + " → imputación → z-score → " + family,
                  row.f1_macro_std)
    row = original[original.name.str.startswith("svm_linear")].sort_values("f1_macro_mean", ascending=False).iloc[0]
    artifact = joblib.load(ROOT / "outputs" / meta["artifacts"][row['name']])
    add_saved("selected_svm_linear", "SVM · lineal", "SVM", "Proyecto base", artifact["pipeline"], row.f1_macro_mean,
              artifact['spec']['params'], "11 mediciones + 3 cocientes → imputación → z-score → SVM lineal", row.f1_macro_std)
    entries.sort(key=lambda r: r["f1_macro"], reverse=True)
    best = entries[0]
    with threadpool_limits(limits=1):
        predictions = joblib.load(ROOT / "outputs" / best["artifact"])["pipeline"].predict(test[FEATURES]).astype(int)
    final = pd.DataFrame({"id": test.id, "prediction": predictions})
    validate_submission(final, test)
    backup = ROOT / "outputs/submission_base.csv"
    if not backup.exists():
        shutil.copy2(ROOT / "entregas/submission_Grupo_1.csv", backup)
    final.to_csv(ROOT / "entregas/submission_Grupo_1.csv", index=False)
    comparison = {
        "sha256": fingerprints(), "winner_id": best["id"], "models": entries,
        "source_winners": [table[table.source == origin].iloc[0].where(pd.notna(table.iloc[0]), None).to_dict()
                           for origin in ["Proyecto base", "Versión alternativa", "Versión Alejo"]],
        "total_configurations": len(table), "validation": "5-fold StratifiedKFold, shuffle=True, random_state=42",
        "selection": "Mayor F1 macro de decisión estándar en resultados ya calculados. Sin búsqueda nueva ni ensamble no validado.",
        "threshold_note": "Los resultados con ajuste de umbral se muestran por separado; no se usan para ordenar. La calibración de la fuente no equivale a una CV anidada estricta.",
        "limits": "La comparación reutiliza CV de selección; diferencias pequeñas no prueban superioridad estadística. Test oculto desconocido.",
        "import_verification": {"source_submission_reproduced": True,
                                "raw_predict_differences_from_source_wrapper": int(np.sum(predictions != source_predictions)),
                                "version_warnings": sorted({str(item.message) for item in caught})},
        "winner_confusion_matrix": aw['confusion_matrix'], "winner_fold_scores": aw['fold_scores'],
        "previous_winner_f1": meta['best_cv']['f1_macro_mean'],
    }
    (ROOT / "outputs/consolidated.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Comparadas {len(table)} configuraciones. Selección: {best['name']} F1={best['f1_macro']:.4f}. Sin entrenar.")


if __name__ == "__main__":
    main()
