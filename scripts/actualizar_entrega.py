"""Actualiza README y evidencia del notebook sin volver a entrenar modelos."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "outputs/consolidated.json").read_text(encoding="utf-8"))
    rows = "\n".join(f"| {m['name']} | {m['source']} | {m['f1_macro']:.4f} |" for m in data['models'])
    readme = f'''# WineLab · Hackathon de calidad del vino

Grupo 1 · Juan Alejandro Cardona, Javier Urrego y Santiago Osorio

Métodos y Modelos Matemáticos para IA · Pontificia Universidad Javeriana Cali

Aplicación Streamlit para explorar datos, comparar modelos y simular la calidad de un
vino con once mediciones fisicoquímicas. Se consolidaron **116 configuraciones ya
evaluadas en tres proyectos**, sin repetir sus búsquedas ni añadir un ensamble sin validar.

## Mejor modelo y parámetros

**k-NN con coseno, 23 vecinos y votos ponderados por distancia: F1 macro CV 0.8624.**
Se reutiliza el pipeline guardado de la versión Alejo. Su decisión estándar fue la mejor
media disponible: 0.8624 frente a 0.8510 del proyecto base y 0.8490 de la otra versión.

| Parámetro | Valor |
|---|---|
| Vecinos | 23 |
| Disimilitud | coseno |
| Votación | `weights="distance"` |
| Entrada | 11 mediciones; id excluido |
| Feature engineering | 6 derivadas + 5 logaritmos: 22 coordenadas |
| Centroides | k-means con 2 y 6 grupos; semilla 42 |
| Coordenadas finales | 40, tras distancias L2/coseno y reescalado |
| PCA final | No |
| Decisión | `pipeline.predict`, sin ajustar el umbral |

La media es 0.862372; desviación muestral entre cinco folds: 0.0183. El test oculto no
tiene etiquetas disponibles. La superioridad estadística no se demuestra por comparar
medias próximas; los resultados proceden de selección con CV y pueden ser optimistas.

## Ejecutar

En el equipo preparado, abrir **`iniciar_app.bat`**. Desde la raíz:

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app/app.py
```

Abrir http://localhost:8501. Para instalar en otro equipo con Python 3.13:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m streamlit run app/app.py
```

La app utiliza un tema oscuro con acentos vino, violeta y cian, inspirado en la versión
Alejo. Funciona sin fuentes o recursos visuales de internet.

- **Mejor modelo:** panel destacado con F1, parámetros exactos y transformaciones.
- **Comparativa:** gráfico y tabla por método, ganadores de las tres versiones y las
  116 configuraciones con procedencia y criterio de comparación. Ranking completo
  visible directamente, colores por familia y ganador en rojo; tabla de F1 con barras.
- **Simulador modular:** escoger k-NN, k-means, SVM lineal/polinomial/RBF o MLP;
  introducir las once mediciones y predecir con el pipeline seleccionado. PCA se
  presenta como transformación acompañada de un clasificador.
- **EDA y PCA:** espectro de autovalores, barras de varianza individual y curva acumulada,
  líneas del 90%/95% y tabla por componente, visibles sin desplegables. Permite alternar
  entre 11 mediciones originales (7/9 componentes para 90%/95%) y 22 atributos químicos.
  Incluye además la distribución de clases.
- **Recorrido técnico:** seis módulos conservan la progresión de la guía.
- **Descargar predicciones:** CSV generado con el ganador consolidado.

## Comparación seleccionada

| Modelo | Procedencia | F1 macro estándar |
|---|---|---:|
{rows}

Resultados completos: [project_comparison.csv](outputs/project_comparison.csv).
Selección, parámetros y trazabilidad: [consolidated.json](outputs/consolidated.json).
El experimento original de 79 candidatos se conserva en `outputs/experiment.json` y
`outputs/cv_results.csv`; documenta el proyecto base, no el ganador final consolidado.

Todos los proyectos usan los mismos valores, columnas y orden de los CSV, cinco folds
estratificados, mezcla y semilla 42. Los resultados de Alejo con umbral ajustado se
conservan en una columna separada, pero **el ranking usa F1 de decisión estándar**.
Su procedimiento de umbral OOF no constituye una CV anidada estricta y no se utiliza
para declarar el ganador. La dispersión ajustada no se atribuye a los scores estándar.
Las métricas proceden de ejecuciones existentes; no se recalcularon por observar las
gráficas. Los tiempos de equipos distintos no se usan como criterio de selección.
La app muestra el entrenamiento medio registrado en segundos por fold en el panel
ganador, la tabla del ranking y los detalles de las barras; no recalcula esos tiempos.

## Datos, prevención de fuga y límites

- Train: 5.197 vinos, 4.369 estándar y 828 superiores; once predictores y target.
- Test: 1.300 vinos, sin target. `id` alinea la entrega y nunca es predictor.
- Los pipelines ajustan escalado, centroides y PCA con el entrenamiento de cada fold.
- Se conserva el desbalance original y F1 macro pondera igual ambas clases.
- Hay 807 filas repetidas por perfil. El diagnóstico agrupado **0.8041** conservado
  en el experimento base corresponde al antiguo k-NN k=11, no al nuevo ganador.
  No se inventa una validación agrupada del pipeline consolidado.
- La figura PCA de la app es descriptiva del proyecto base, de 14 variables;
  no es una proyección del clasificador final de 40 coordenadas.
- La comparación entre versiones selecciona sobre CV existente; no es test independiente.

## Modelos guardados: sin entrenar durante la exposición

`outputs/selected_models/` contiene los seis representantes finales, con su pipeline
completo. La app los carga de disco y usa caché en memoria. Cerrar y abrir la app no
obliga a entrenar. Las 79 configuraciones base también se conservan preparadas en
`outputs/demo_models/`. Configuraciones personalizadas inéditas pueden necesitar un
primer ajuste; el botón de evaluación de cinco folds vuelve a entrenar.

Se comprobó que el artefacto importado reproduce exactamente las 1.300 predicciones
de su entrega de origen. La versión estándar del ganador produce esas mismas
predicciones. El CSV anterior está archivado en `outputs/submission_base.csv`.

## Entregables

- [Notebook ejecutado](notebooks/01_eda_y_modelado.ipynb): progresión del proyecto base
  y consolidación final con resultados existentes, sin entrenamiento adicional.
- [Aplicación](app/app.py): comparación, ganador y simulación modular.
- [submission.csv](entregas/submission.csv): exactamente `id,prediction`, 1.300 registros
  más cabecera, sin nulos, clases 0/1 y orden original de test.
- [Presentación de clase](presentacion/PRESENTACION_CLASE.html): diapositivas autónomas.

## Verificación y reproducción de predicciones

```powershell
.\\.venv\\Scripts\\python.exe -m pytest tests -q
.\\.venv\\Scripts\\python.exe scripts/verificar_entrega.py
```

La consolidación se implementa en `scripts/consolidar_proyectos.py`; requiere las
carpetas originales aportadas por el equipo para extraer su evidencia. La app y la
verificación usan los artefactos incluidos y no necesitan esas carpetas. Las clases
reutilizadas se incluyen en `src/pipeline_wine.py`; no se ejecuta su entrenamiento
al importar. Dependencias fijadas en `requirements.txt` y `requirements.lock.txt`.

Solo se utilizan k-NN, k-means, PCA, SVM y MLP, conforme a la guía. No se emplean
árboles, boosting ni AutoML. La inspección visual automatizada del navegador no estuvo
disponible; navegación, formularios y predicciones se verifican con Streamlit AppTest.
'''
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    path = ROOT / "notebooks/01_eda_y_modelado.ipynb"
    nb = nbf.read(path, as_version=4)
    nb.cells = [c for c in nb.cells if not c.metadata.get('consolidated_final')]
    for c in nb.cells:
        if c.cell_type == 'markdown':
            c.source = c.source.replace('docs/grupo/EXPLICACION.html', 'README.md').replace('docs/grupo/GUION_EXPOSICION.html', 'presentacion/PRESENTACION_CLASE.html')
            c.source = c.source.replace('`CONTEXT.md` y `AGENTS.md`', '`outputs/consolidated.json`')
            if '### Aplicación y defensa' in c.source:
                c.source = '''### Aplicación y entregables
Abrir `app/app.py` con Streamlit. La app contiene mejor modelo, comparativa de tres
versiones, simulador modular, EDA/PCA y recorrido técnico por los seis pasos.
README.md documenta ejecución, resultados y parámetros; la última sección de este
cuaderno comprueba el CSV de la selección consolidada.'''
        elif "saved = pd.read_csv(ROOT/'entregas/submission.csv')" in c.source:
            c.source = c.source.replace("ROOT/'entregas/submission.csv'", "ROOT/'outputs/submission_base.csv'")
    if '**Actualización:**' not in nb.cells[0].source:
        nb.cells[0].source += '\n\n**Actualización:** esta primera parte documenta la progresión base. La selección final entre las tres versiones y el CSV vigente están en la última sección.'
    marker = nbf.v4.new_markdown_cell('''## Selección final entre tres proyectos

Se reutilizan las 116 métricas existentes (79 + 22 + 15), sin nuevas búsquedas. Se
comparan decisiones estándar y se separa el ajuste de umbral. El ganador es k-NN
k=23 con coseno, votos por distancia y centroides 2/6. No utiliza PCA. Los resultados
anteriores del cuaderno son históricos del proyecto base; el CSV vigente se verifica aquí.''')
    marker.metadata['consolidated_final'] = True
    code = nbf.v4.new_code_cell('''from pathlib import Path
import sys, json
ROOT = Path.cwd()
if not (ROOT/'src').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import joblib
from IPython.display import display
from src.data import load_data, FEATURES, validate_submission
train, test = load_data()
final = json.loads((ROOT/'outputs/consolidated.json').read_text(encoding='utf-8'))
comparison = pd.read_csv(ROOT/'outputs/project_comparison.csv')
display(comparison[['source','name','f1_macro','reported_f1','adjusted_threshold']].head(15))
winner = next(m for m in final['models'] if m['id'] == final['winner_id'])
display(pd.Series({'Modelo': winner['name'], 'F1 macro estándar': winner['f1_macro'], 'Parámetros': winner['params']}))
pipe = joblib.load(ROOT/'outputs'/winner['artifact'])['pipeline']
submission = pd.read_csv(ROOT/'entregas/submission.csv')
assert validate_submission(submission, test)
np.testing.assert_array_equal(pipe.predict(test[FEATURES]), submission.prediction)
display(submission.head())
print('Entrega final comprobada: 1.300 predicciones del ganador consolidado. Sin reentrenar.')''')
    code.metadata['consolidated_final'] = True
    # Ejecutar solo la celda nueva; las salidas anteriores ya fueron ejecutadas y se conservan.
    scratch = nbf.v4.new_notebook(cells=[code], metadata=nb.metadata)
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    import os
    state = ROOT / "outputs/.jupyter"
    (state/'kernels/hackathon').mkdir(parents=True, exist_ok=True)
    (state/'runtime').mkdir(parents=True, exist_ok=True)
    (state/'kernels/hackathon/kernel.json').write_text(json.dumps({
        'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Hackathon local','language':'python'}),encoding='utf-8')
    os.environ['IPYTHONDIR'] = str(state/'ipython')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(state/'runtime')
    km = KernelManager(kernel_name='hackathon', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(state/'kernels')]))
    try:
        NotebookClient(scratch, km=km, timeout=90, resources={'metadata': {'path':str(ROOT)}}).execute()
    finally:
        if km.has_kernel: km.shutdown_kernel(now=True)
    executed = scratch.cells[0]
    executed.execution_count = max(c.execution_count or 0 for c in nb.cells if c.cell_type == 'code') + 1
    nb.cells.extend([marker, executed])
    nbf.write(nb, path)
    print('README y notebook actualizados; consolidación ejecutada sin ajustar modelos.')


if __name__ == '__main__':
    main()
