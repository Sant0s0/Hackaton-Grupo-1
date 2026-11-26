"""Construye y ejecuta el notebook solicitado, sin inventar salidas."""
from pathlib import Path
import argparse
import json
import os
import sys
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def create():
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))

    md("""# Hackathon: calidad del vino
**Métodos y Modelos Matemáticos para IA · Javeriana Cali**

Este cuaderno sigue los **seis pasos de la guía oficial**, páginas 3–4. Usa solo
los modelos permitidos y los CSV entregados. Las salidas se calculan ejecutando
código; los resultados de la búsqueda completa se recuperan de archivos auditables.

La métrica oficial es **F1 macro**. No conocemos las etiquetas del test oculto.
El documento `docs/EXPLICACION.html` desarrolla las decisiones en lenguaje sencillo.
""")
    md("""## 1. Configuración y roles (0–10 min)
Responsabilidades propuestas: datos y fundamentos; modelos y validación; aplicación
y presentación. Los nombres y las contribuciones reales corresponden al equipo.

Cada vino es un vector de once coordenadas. `id` solo alinea la entrega y `target`
es la etiqueta binaria. Semilla 42 y los mismos cinco folds para todas las alternativas.
""")
    code("""from pathlib import Path
import sys, json, io
ROOT = Path.cwd()
if not (ROOT / 'src').exists():
    ROOT = ROOT.parent
assert (ROOT / 'src').exists(), 'Ejecuta desde la raíz o notebooks/'
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.data import FEATURES, SEED, load_data, audit_data, validate_submission, fingerprints
from src.models import candidates, build_pipeline
from src.evaluation import evaluate_candidate, run_experiment
from src.plots import class_figure, correlation_figure, distribution_figure, pca_figure, confusion_figure
def show_fig(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    display(Image(data=buffer.getvalue()))
    plt.close(fig)
train, test = load_data()
X, y = train[FEATURES], train.target.astype(int)
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
assert 'id' not in X.columns and 'target' not in X.columns
display(pd.DataFrame({'Archivo': ['train_public.csv', 'test_public_features.csv'],
                      'Filas': [len(train), len(test)], 'Columnas': [train.shape[1], test.shape[1]]}))
""")
    md("""## 2. EDA y limpieza (10–30 min)
Se inspeccionan desbalance, faltantes, extremos IQR, escalas y correlaciones.
Los outliers son banderas descriptivas: no se borran por ser extremos. La imputación
por mediana queda en el pipeline para aprenderla únicamente con el entrenamiento.
""")
    code("""audit = audit_data(train, test)
display(pd.Series({k:v for k,v in audit.items() if k not in ['iqr_flags_by_feature','sha256','class_counts']}))
display(train[FEATURES].describe().T)
display(pd.DataFrame({'Banderas IQR': audit['iqr_flags_by_feature']}))
show_fig(class_figure(train))
show_fig(correlation_figure(train))
show_fig(distribution_figure(train))
""")
    code("""# Por qué accuracy no basta: predecir siempre cero ignora por completo la clase superior.
all_zero = np.zeros(len(y), dtype=int)
display(pd.Series({'Accuracy siempre 0 (referencia aritmética)': accuracy_score(y, all_zero),
                   'F1 macro siempre 0 (referencia aritmética)': f1_score(y, all_zero, average='macro')}))
# No se usa un clasificador adicional fuera de los métodos permitidos.
display(Markdown(f"Hay **{audit['duplicate_feature_rows_train']} filas repetidas** por sus 11 mediciones. "
                 "Se añade una validación agrupada como control de generalización."))
""")
    md(r"""## 3. Baseline obligatorio (30–50 min)
Se vuelve a ejecutar aquí la rejilla base completa: **k=3,5,7 × Euclidiana/Manhattan**,
más k-means con asignación por mayoría. Cada fold ajusta su propio escalador.

\(d_p(x,z)=(\sum_j |x_j-z_j|^p)^{1/p}\): Manhattan usa p=1, Euclidiana p=2.
k-NN memoriza y vota; k-means minimiza la suma de distancias cuadráticas a centroides.
La clase de un clúster se aprende solo con las etiquetas del entrenamiento del fold.
""")
    code("""specs = candidates('complete')
baseline_rows = []
for spec in [s for s in specs if s.stage == 3]:
    summary, detail, oof = evaluate_candidate(spec, X, y, splits)
    baseline_rows.append(summary)
baseline_results = pd.DataFrame(baseline_rows).sort_values('f1_macro_mean', ascending=False)
display(baseline_results[['name','f1_macro_mean','f1_macro_std','f1_1_mean','fit_seconds_mean','prediction_seconds_mean']])
""")
    md("""## 4. Variables nuevas y PCA (50–75 min)
Se conservan los once atributos y se agregan tres cocientes antes del escalado:

1. Acidez fija / volátil: balance relativo entre formas de acidez.
2. SO₂ libre / total: fracción del dióxido de azufre que está libre.
3. Alcohol / densidad: relación empírica entre dos propiedades relacionadas.

Son hipótesis de representación, no relaciones causales demostradas. Denominadores
casi nulos generan NaN y se imputan dentro del fold. También se ensayan distancias
a ocho centroides aprendidos únicamente con el entrenamiento.
""")
    code("""from src.features import ChemicalRatios, RATIO_NAMES
ratios = ChemicalRatios().fit_transform(X)
display(pd.DataFrame(ratios[:5], columns=FEATURES + RATIO_NAMES))
assert ratios.shape[1] == 14
show_fig(pca_figure(train))
""")
    md("""### Conexión con el ejemplo de PCA de clase
Se reproduce el conjunto de cuatro estudiantes de `PCA_TeoríaConEjemplo.html`.
La clase divide covarianza por N; scikit-learn informa varianzas con N−1.
Ese factor común cambia autovalores, pero conserva direcciones y porcentajes.
Los signos de los autovectores pueden invertirse sin cambiar el subespacio.
""")
    code("""toy = np.array([[1,3],[2,2],[4,6],[5,5]], dtype=float)
centered = toy - toy.mean(axis=0)
covariance = centered.T @ centered / len(toy)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
order = np.argsort(eigenvalues)[::-1]
eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
display(pd.DataFrame(covariance))
display(pd.DataFrame({'autovalor': eigenvalues, 'fracción de varianza': eigenvalues/eigenvalues.sum()}))
display(pd.DataFrame(centered @ eigenvectors[:, :1], columns=['PC1']))
np.testing.assert_allclose(eigenvalues, [4.5, 0.5])
""")
    code("""# Ablación controlada: k=5 y Manhattan permanecen fijos.
ablation_rows = [next(row for row in baseline_rows if row['name'] == 'base_knn_k5_manhattan')]
for spec in [s for s in specs if s.name in ['fe_knn_k5_manhattan',
             'pca0.9_knn_k5_manhattan', 'pca0.95_knn_k5_manhattan']]:
    summary, _, _ = evaluate_candidate(spec, X, y, splits)
    ablation_rows.append(summary)
ablation = pd.DataFrame(ablation_rows)
display(ablation[['name','f1_macro_mean','f1_macro_std','dimensions_mean','fit_seconds_mean','prediction_seconds_mean']])
display(Markdown('PCA se decide por su resultado y tiempo medidos; retener varianza no garantiza conservar la separación entre clases.'))
""")
    md("""## 5. SVM y MLP (75–100 min)
**SVM:** se comparan kernels lineal, polinomial y RBF; C regula la penalización por
violaciones y gamma el alcance de RBF. Se compara `class_weight=None/balanced`.

**MLP:** una o dos capas, ReLU/tanh y alpha. La red minimiza entropía cruzada
regularizada mediante Adam; backpropagation calcula gradientes. No minimiza F1
directamente. Se prueba ponderación de clases calculada solo dentro del fold.

El registro `src/models.py` contiene toda la rejilla explícita. No se utilizan árboles,
boosting ni AutoML. Para repetir los 79 candidatos cambia `REENTRENAR_TODO=True`.
Si faltan los resultados completos, se generan automáticamente.
""")
    code("""REENTRENAR_TODO = False
if REENTRENAR_TODO or not (ROOT/'outputs/experiment.json').exists():
    results, metadata = run_experiment('complete')
else:
    results = pd.read_csv(ROOT/'outputs/cv_results.csv')
    metadata = json.loads((ROOT/'outputs/experiment.json').read_text(encoding='utf-8'))
assert metadata['sha256'] == fingerprints(), 'Los resultados pertenecen a otros datos'
assert metadata['profile'] == 'complete' and metadata['folds'] == 5
display(results[['name','family','f1_macro_mean','f1_macro_std','f1_1_mean','fit_seconds_mean']])
""")
    code("""# Reejecución representativa: mejor SVM de cada kernel y mejor MLP.
selected_names = []
for kernel in ['linear', 'poly', 'rbf']:
    names = [s.name for s in specs if s.family == 'SVM' and s.params['kernel'] == kernel]
    selected_names.append(results[results.name.isin(names)].iloc[0]['name'])
selected_names.append(results[results.family == 'MLP'].iloc[0]['name'])
advanced_check = []
for name in selected_names:
    spec = next(s for s in specs if s.name == name)
    summary, _, _ = evaluate_candidate(spec, X, y, splits)
    expected = float(results.loc[results.name == name, 'f1_macro_mean'].iloc[0])
    np.testing.assert_allclose(summary['f1_macro_mean'], expected, atol=1e-12)
    advanced_check.append(summary)
display(pd.DataFrame(advanced_check)[['name','f1_macro_mean','f1_macro_std','f1_1_mean','warnings_count']])
""")
    md("""## 6. Selección, Streamlit, submission y pitch (100–120 min)
Se elige el mayor F1 macro medio de los mismos cinco folds. En empate se compara
menor desviación y nombre. El ganador puede ser un método sencillo: avanzar hacia
modelos más complejos significa evaluarlos, no asumir que ganan.

La CV se utiliza para seleccionar hiperparámetros; su máximo puede ser optimista.
La desviación entre folds no es un intervalo de confianza. El control por perfiles
también es diagnóstico del ganador ya elegido, no una evaluación anidada.
""")
    code("""display(pd.Series(metadata['best']))
display(pd.Series({'F1 macro CV': metadata['best_cv']['f1_macro_mean'],
                   'Desviación entre folds': metadata['best_cv']['f1_macro_std'],
                   'F1 macro control por perfiles': metadata['grouped_diagnostic']['f1_macro_mean'],
                   'F1 test oculto': metadata['hidden_test_f1_macro']}))
show_fig(confusion_figure(metadata['oof_confusion_matrix']))
display(pd.read_csv(ROOT/'outputs/grouped_validation_folds.csv'))
""")
    code("""# Reproducir y verificar la entrega usando el modelo final ajustado con TODO train.
import joblib
artifact = joblib.load(ROOT/'outputs'/metadata['artifacts'][metadata['best']['name']])
with threadpool_limits(limits=1):
    prediction = artifact['pipeline'].predict(test[FEATURES]).astype(int)
submission = pd.DataFrame({'id': test.id, 'prediction': prediction})
assert validate_submission(submission, test)
saved = pd.read_csv(ROOT/'entregas/submission.csv')
pd.testing.assert_frame_equal(saved, submission)
assert len(saved) == 1300 and list(saved.columns) == ['id', 'prediction']
display(saved.head(10))
display(saved.prediction.value_counts().rename('Cantidad de predicciones'))
print('Verificado: 1.300 observaciones + cabecera = 1.301 líneas; orden de id conservado.')
""")
    md("""### Aplicación y defensa
Ejecutar desde la raíz:
```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app/app.py
```
La aplicación tiene los seis pasos, selector de todos los métodos permitidos,
visualizaciones EDA/PCA, tabla comparativa y simuladores. Las configuraciones
personalizadas se mantienen separadas de la entrega oficial guardada.

Documentos: `README.md` (ejecución y resultados), `docs/EXPLICACION.html` (cada punto
y su justificación), `docs/PITCH.html` (guion de 5 min), `CONTEXT.md` y `AGENTS.md`
(continuidad para IA). Las fuentes del curso y páginas se indican en el documento.

La publicación en GitHub y el historial colaborativo deben reflejar el trabajo real
del equipo; este cuaderno no inventa autores, commits ni resultados del leaderboard.
""")
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (Hackathon)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]}})
    path = ROOT / "notebooks/01_eda_y_modelado.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    return nb, path


def execute(nb, path):
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    # Kernel local al proyecto: no modifica las configuraciones personales.
    state = ROOT / "outputs/.jupyter"
    specdir = state / "kernels/hackathon"
    specdir.mkdir(parents=True, exist_ok=True)
    (specdir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Hackathon local", "language": "python"}), encoding="utf-8")
    os.environ["IPYTHONDIR"] = str(state / "ipython")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(state / "runtime")
    (state / "runtime").mkdir(parents=True, exist_ok=True)
    ksm = KernelSpecManager(kernel_dirs=[str(state / "kernels")])
    km = KernelManager(kernel_name="hackathon", kernel_spec_manager=ksm,
                       connection_file=str(state / "runtime/connection.json"))
    client = NotebookClient(nb, km=km, timeout=600, resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        nbf.write(nb, path)
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    errors = [out for cell in nb.cells if cell.cell_type == "code" for out in cell.outputs if out.output_type == "error"]
    if errors:
        raise RuntimeError(errors)
    print(f"Notebook ejecutado: {path.name}; {sum(c.cell_type == 'code' for c in nb.cells)} celdas de código sin errores.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    notebook, path = create()
    if args.execute:
        execute(notebook, path)
    else:
        print(path)
