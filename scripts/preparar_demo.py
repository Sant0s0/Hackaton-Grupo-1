"""Prepara todos los ajustes evaluados para presentar sin entrenar en vivo."""
from pathlib import Path
import json
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sklearn
from src.data import ROOT, fingerprints, load_data
from src.models import Candidate
from src.persistence import model_path, prepare_model


def main():
    train, _ = load_data()
    hashes = fingerprints()
    meta = json.loads((ROOT / "outputs/experiment.json").read_text(encoding="utf-8"))
    if meta["sha256"] != hashes:
        raise ValueError("Los resultados no corresponden a los datos actuales. Actualiza el experimento primero.")
    specs = [Candidate(**item) for item in json.loads((ROOT / "outputs/candidates.json").read_text(encoding="utf-8"))]
    manifest = {"sha256": hashes, "sklearn_version": sklearn.__version__, "models": {}}
    start = perf_counter()
    for index, spec in enumerate(specs, 1):
        artifact = prepare_model(spec, train, hashes, meta)
        manifest["models"][spec.name] = {
            "file": model_path(spec, hashes["train_public.csv"]).name,
            "spec": artifact["spec"], "fit_warnings": artifact["fit_warnings"]}
        print(f"[{index}/{len(specs)}] Listo: {spec.name}", flush=True)
    manifest["preparation_seconds"] = perf_counter() - start
    (ROOT / "outputs/demo_models/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(specs)} configuraciones guardadas. Métricas y submission originales conservados.")


if __name__ == "__main__":
    main()
