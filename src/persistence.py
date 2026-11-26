"""Archivos de modelos completos, separados de las métricas de validación."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings
import joblib
import sklearn
from threadpoolctl import threadpool_limits
from src.data import FEATURES, ROOT
from src.models import build_pipeline, candidates


def canonical_spec(spec):
    return json.loads(json.dumps(spec.to_dict(), sort_keys=True))


def model_path(spec, train_hash, directory=None):
    payload = {"spec": canonical_spec(spec), "train_sha256": train_hash,
               "sklearn": sklearn.__version__, "format": 1}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    if directory is not None:
        return Path(directory) / f"model_{key}.joblib"
    registered = any(canonical_spec(s) == canonical_spec(spec) for s in candidates("complete"))
    return (ROOT / "outputs/demo_models" / f"model_{key}.joblib" if registered
            else ROOT / "outputs" / f"app_{key}.joblib")


def load_saved(spec, hashes, meta=None, directory=None):
    paths = [model_path(spec, hashes["train_public.csv"], directory)]
    if directory is None and meta and spec.name in meta.get("artifacts", {}):
        paths.append(ROOT / "outputs" / meta["artifacts"][spec.name])
    for path in paths:
        if not path.is_file():
            continue
        artifact = joblib.load(path)
        if (artifact.get("spec") == canonical_spec(spec)
                and artifact.get("sha256", {}).get("train_public.csv") == hashes["train_public.csv"]
                and artifact.get("sklearn_version") == sklearn.__version__):
            return artifact
    return None


def prepare_model(spec, train, hashes, meta=None, directory=None):
    """Carga un ajuste existente o aprende UNA vez sobre train y lo guarda.

    No calcula nuevos scores de CV ni modifica ganador, métricas o submission.
    """
    path = model_path(spec, hashes["train_public.csv"], directory)
    artifact = load_saved(spec, hashes, meta, directory)
    needs_save = artifact is None or not path.exists()
    if artifact is None:
        pipeline = build_pipeline(spec)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            pipeline.fit(train[FEATURES], train.target)
        artifact = {"pipeline": pipeline, "spec": canonical_spec(spec), "sha256": hashes,
                    "sklearn_version": sklearn.__version__,
                    "fit_warnings": sorted({str(item.message) for item in caught})}
    if needs_save:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
        try:
            joblib.dump(artifact, temporary, compress=3)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return artifact
