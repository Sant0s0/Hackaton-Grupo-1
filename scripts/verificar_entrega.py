"""Valida los archivos obligatorios y la coherencia de la entrega completa."""
from pathlib import Path
import json
import sys
from html.parser import HTMLParser
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nbformat
import numpy as np
import pandas as pd
from src.data import ROOT, FEATURES, load_data, fingerprints, validate_submission
from src.models import candidates


def main():
    required = ["notebooks/01_eda_y_modelado.ipynb", "app/app.py", "entregas/submission.csv",
                "README.md", "CONTEXT.md", "AGENTS.md", "docs/grupo/EXPLICACION.html", "docs/grupo/GUION_EXPOSICION.html",
                "presentacion/PRESENTACION_CLASE.html"]
    for name in required:
        path = ROOT / name
        assert path.is_file() and path.stat().st_size > 0, f"Falta {name}"
    train, test = load_data()
    submission_path = ROOT / "entregas/submission.csv"
    submission = pd.read_csv(submission_path)
    assert validate_submission(submission, test)
    assert len(submission) == 1300
    assert len(submission_path.read_text(encoding="utf-8").splitlines()) == 1301
    reject_constant = lambda value: (_ for _ in ()).throw(ValueError(f"JSON no estándar: {value}"))
    meta = json.loads((ROOT / "outputs/experiment.json").read_text(encoding="utf-8"), parse_constant=reject_constant)
    assert meta["sha256"] == fingerprints()
    assert meta["profile"] == "complete" and meta["folds"] == 5
    results = pd.read_csv(ROOT / "outputs/cv_results.csv")
    folds = pd.read_csv(ROOT / "outputs/cv_folds.csv")
    assignments = pd.read_csv(ROOT / "outputs/fold_assignments.csv")
    assert set(results.name) == {spec.name for spec in candidates("complete")}
    assert len(results) == meta["candidates"]
    assert len(folds) == len(results) * 5
    assert len(assignments) == len(train) and assignments.id.equals(train.id)
    assert set(assignments.fold) == {1, 2, 3, 4, 5}
    np.testing.assert_allclose(results.set_index("name").f1_macro_mean.sort_index(),
                               folds.groupby("name").f1_macro.mean().sort_index())
    assert results.iloc[0]["name"] == meta["best"]["name"]
    nb = nbformat.read(ROOT / required[0], as_version=4)
    nbformat.validate(nb)
    code = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert code and all(cell.execution_count is not None for cell in code), "Notebook sin ejecutar"
    assert [cell.execution_count for cell in code] == list(range(1, len(code)+1))
    assert not [out for cell in code for out in cell.outputs if out.output_type == "error"]
    pngs = [out for cell in code for out in cell.outputs if "image/png" in out.get("data", {})]
    assert len(pngs) >= 5, "El notebook debe incluir las gráficas renderizadas, no solo texto Figure(...)"
    for name in ["docs/grupo/EXPLICACION.html", "docs/grupo/GUION_EXPOSICION.html", "README.md"]:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert all(person in text for person in ["Juan Alejandro Cardona", "Javier Urrego", "Santiago Osorio"])
        assert "Hackaton-Grupo-1" in text
        assert "\ufffd" not in text, f"Carácter de sustitución en {name}"
    presentation = (ROOT / "presentacion/PRESENTACION_CLASE.html").read_text(encoding="utf-8")
    class Slides(HTMLParser):
        def __init__(self):
            super().__init__()
            self.count = 0
        def handle_starttag(self, tag, attrs):
            if tag == "section" and "slide" in dict(attrs).get("class", "").split():
                self.count += 1
    slides = Slides()
    slides.feed(presentation)
    assert slides.count == 8
    assert "docs/grupo" not in presentation and "GUION_EXPOSICION" not in presentation
    assert "https://cdn" not in presentation, "Las diapositivas deben funcionar sin internet"
    manifest = json.loads((ROOT / "outputs/demo_models/manifest.json").read_text())
    assert manifest["sha256"] == fingerprints()
    assert set(manifest["models"]) == set(results.name)
    assert all((ROOT / "outputs/demo_models" / entry["file"]).is_file()
               for entry in manifest["models"].values())
    print(f"Entrega verificada: {len(required)} archivos principales; {len(results)} candidatos × 5 folds; "
          f"{len(code)} celdas ejecutadas; 1.300 predicciones + cabecera; hashes y orden de id correctos.")
    print("El F1 del test oculto y la evaluación oral quedan a cargo de la profesora.")


if __name__ == "__main__":
    main()
