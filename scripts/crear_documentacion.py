"""README e informes autónomos con resultados medidos, no números de ejemplo."""
from pathlib import Path
import base64
from html import escape
import io
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import pandas as pd
from src.data import ROOT, FEATURES, load_data
from src.plots import class_figure, correlation_figure, distribution_figure, pca_figure, confusion_figure

TEAM = ["Juan Alejandro Cardona", "Javier Urrego", "Santiago Osorio"]
REPO = "https://github.com/Sant0s0/Hackaton-Grupo-1"
CSS = """
:root{color-scheme:light;--ink:#21364a;--rose:#96375c;--line:#d8e1e9}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#eef2f6;color:var(--ink);font:17px/1.7 'Segoe UI',Arial,sans-serif}
main{max-width:1060px;margin:35px auto;background:white;padding:55px 65px;border-top:8px solid var(--rose);border-radius:8px}
h1{font-size:2.7rem;line-height:1.13;letter-spacing:-.045em;margin:16px 0 20px}h2{font-size:1.65rem;line-height:1.3;margin:54px 0 15px;scroll-margin-top:20px}h3{font-size:1.14rem;margin:28px 0 9px}
p{margin:10px 0 18px}.eyebrow{font-weight:700;letter-spacing:.12em;color:var(--rose);font-size:.8rem}.lead{font-size:1.2rem;color:#526779}
a{color:#96375c;text-underline-offset:3px}nav{border:1px solid var(--line);padding:20px 26px;margin:28px 0;border-radius:10px;background:#f8fafc}nav a{display:inline-block;margin:3px 12px 3px 0}
.note,.result{border-left:4px solid #396d99;padding:15px 22px;background:#eef5fa;margin:22px 0;border-radius:0 8px 8px 0}.result{border-color:var(--rose);background:#faf0f4}.muted{color:#587084;font-size:.9rem}
table{border-collapse:collapse;width:100%;margin:18px 0 24px;font-size:.9rem;line-height:1.5}th{background:#eaf0f5;text-align:left;font-weight:700}td,th{padding:12px 13px;border-bottom:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}tr{break-inside:avoid}
code{background:#edf2f6;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere;font-size:.88em}pre{padding:18px;background:#142b3d;color:white;white-space:pre-wrap;border-radius:8px;overflow-wrap:anywhere}pre code{padding:0;background:none;color:inherit}
.formula{font:1.05rem/1.8 'Cambria Math','Segoe UI',serif;background:#f6f8fa;padding:18px 22px;border:1px solid var(--line);border-radius:8px;overflow-wrap:anywhere}
figure{margin:24px 0}figure img{max-width:100%;height:auto}figcaption{font-size:.88rem;color:#587084;margin:6px 0}li{margin:8px 0}.pill{display:inline-block;padding:5px 13px;border:1px solid var(--line);border-radius:20px;margin:4px 8px 4px 0;font-size:.88rem}
footer{margin-top:50px;border-top:1px solid var(--line);padding-top:22px;color:#587084;font-size:.9rem}button{background:var(--rose);color:white;border:0;border-radius:6px;padding:10px 17px;font:inherit;cursor:pointer}
@media(max-width:700px){main{margin:0;padding:30px 20px;border-radius:0}body{font-size:16px}h1{font-size:2.1rem}table{font-size:.8rem}td,th{padding:8px 6px}}
@media print{@page{size:A4;margin:18mm}body{background:white;font-size:10pt;line-height:1.5}main{padding:0;margin:0;border:0;max-width:none}h1{font-size:29pt}h2{font-size:17pt;margin-top:25pt;break-after:avoid}h3{break-after:avoid}table{font-size:8.5pt}figure,.note,.result,.formula{break-inside:avoid}nav,.print{display:none}a{color:inherit}pre{background:#eef2f6;color:#21364a}footer{font-size:8.5pt}}
"""


def shell(title, body):
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(title)}</title><style>{CSS}</style></head><body><main>{body}<footer>Grupo 1 · {' · '.join(TEAM)}<br>Métodos y Modelos Matemáticos para IA · Javeriana Cali · 26 de noviembre de 2026</footer></main></body></html>"""


def figure(fig, caption, name):
    path = ROOT / "outputs/figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=125, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>'


def results_table(frame):
    view = frame[["name", "f1_macro_mean", "f1_macro_std", "f1_1_mean", "fit_seconds_mean", "prediction_seconds_mean"]].copy()
    view.columns = ["Configuración", "F1 macro", "Desv. folds", "F1 superior", "Ajuste s/fold", "Predicción s/fold"]
    return view.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}", escape=True)


def main():
    (ROOT / "docs/grupo").mkdir(parents=True, exist_ok=True)
    train, test = load_data()
    meta = json.loads((ROOT / "outputs/experiment.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "outputs/data_audit.json").read_text(encoding="utf-8"))
    results = pd.read_csv(ROOT / "outputs/cv_results.csv")
    base = results[results.stage == 3].iloc[0]
    winner = results.iloc[0]
    best_family = results.groupby("family", sort=False).head(1)
    table_names = [base["name"], *best_family.name.tolist()]
    for kernel in ["linear", "poly"]:
        table_names.append(results[results.name.str.startswith("svm_" + kernel)].iloc[0]["name"])
    representative = results[results.name.isin(table_names)]
    paired = results[results.name.isin(["base_knn_k5_manhattan", "fe_knn_k5_manhattan", "pca0.9_knn_k5_manhattan", "pca0.95_knn_k5_manhattan"])]
    gain = winner.f1_macro_mean - base.f1_macro_mean
    grouped = meta["grouped_diagnostic"]["f1_macro_mean"]
    warnings_count = int(results.warnings_count.sum())
    tn, fp, fn, tp = [int(value) for row in meta["oof_confusion_matrix"] for value in row]
    class_fig = figure(class_figure(train), "Distribución observada: 4.369 vinos estándar y 828 superiores.", "clases.png")
    corr_fig = figure(correlation_figure(train), "Correlaciones entre las once variables originales; correlación no significa causalidad.", "correlacion.png")
    hist_fig = figure(distribution_figure(train), "Distribuciones por clase: las regiones se superponen y no basta un único umbral.", "distribuciones.png")
    pca_fig = figure(pca_figure(train), "PCA descriptivo sobre train con tres cocientes. Los PCA de validación se ajustan por fold.", "pca.png")
    confusion_fig = figure(confusion_figure(meta["oof_confusion_matrix"]),
        f"Predicciones fuera de fold del ganador: {tn:,} TN, {fp:,} FP, {fn:,} FN y {tp:,} TP.", "confusion.png")
    body = f"""
<div class="eyebrow">GRUPO 1 · MATERIAL INTERNO DE ESTUDIO</div>
<h1>De las mediciones del vino<br>a una decisión de calidad</h1>
<p class="lead">Qué se hizo en cada punto de la guía, por qué se hizo así y cómo se conecta con las clases del semestre.</p>
<div class="note">Este documento es para preparar al equipo. Las diapositivas para la clase están en <a href="../../presentacion/PRESENTACION_CLASE.html">PRESENTACION_CLASE.html</a> y la aplicación se abre por separado. El <a href="GUION_EXPOSICION.html">guion de exposición</a> también es material del grupo.</div>
<p>{' · '.join(TEAM)}</p><p><a href="{REPO}">Repositorio de Grupo 1</a></p>
<p class="print"><button onclick="window.print()">Imprimir o guardar como PDF</button></p>
<div class="result"><strong>Resultado medido:</strong> el mejor candidato de los {meta['candidates']} evaluados fue k-NN con 11 vecinos, disimilitud coseno, votación ponderada por distancia y tres cocientes. F1 macro por CV: <strong>{winner.f1_macro_mean:.4f} ± {winner.f1_macro_std:.4f}</strong>. El puntaje del test oculto todavía no se conoce.</div>
<nav aria-label="Contenido"><strong>Recorrido del documento</strong><br>
<a href="#contexto">1. Problema y datos</a><a href="#reglas">2. Métodos permitidos</a><a href="#pasos">3. Los seis pasos</a><a href="#entregables">4. Entregables</a><a href="#rubrica">5. Rúbrica</a><a href="#uso">Cómo usar la app</a><a href="#fuentes">Conexión con las clases</a><a href="#preguntas">Preguntas de defensa</a></nav>

<h2 id="contexto">1. Contexto y datos: ¿qué estamos intentando resolver?</h2>
<p>El objetivo es clasificar un vino como <strong>calidad estándar (0)</strong> o <strong>calidad superior (1)</strong> a partir de once propiedades de laboratorio. Es aprendizaje supervisado porque las muestras de entrenamiento ya tienen una etiqueta. No predecimos una nota continua ni inventamos el umbral que convirtió la calidad original en binaria: ese umbral no aparece en la guía.</p>
<p>El archivo de entrenamiento tiene 5.197 filas. El de prueba tiene 1.300 y no contiene etiquetas. Este último sirve para producir predicciones; no permite calcular F1, accuracy o una matriz de confusión reales. El <code>id</code> identifica y ordena las muestras, pero no es una propiedad fisicoquímica y se excluye de los predictores.</p>
<h3>¿Por qué F1 macro?</h3>
<p>Hay {train.target.eq(0).mean():.2%} de clase 0 y {train.target.eq(1).mean():.2%} de clase 1. Si siempre respondiéramos «estándar», acertaríamos aproximadamente el 84% sin detectar ningún vino superior. Accuracy ocultaría ese fallo. F1 combina precisión y recall; la versión macro da el mismo peso a cada clase.</p>
<div class="formula">Precisión = TP / (TP + FP)<br>Recall = TP / (TP + FN)<br>F1 de una clase = 2TP / (2TP + FP + FN)<br>F1 macro = (F1 de clase 0 + F1 de clase 1) / 2</div>
{class_fig}
<h3>Diccionario resumido</h3>
<table><tr><th>Variable</th><th>Qué representa</th><th>Unidad de la guía</th></tr>
<tr><td>fixed_acidity</td><td>Acidez fija, principalmente tartárica</td><td>g/dm³</td></tr>
<tr><td>volatile_acidity</td><td>Acidez volátil, asociada al ácido acético</td><td>g/dm³</td></tr>
<tr><td>citric_acid</td><td>Contenido de ácido cítrico</td><td>g/dm³</td></tr>
<tr><td>residual_sugar</td><td>Azúcar que queda tras la fermentación</td><td>g/dm³</td></tr>
<tr><td>chlorides</td><td>Concentración de sal</td><td>g/dm³</td></tr>
<tr><td>free_sulfur_dioxide</td><td>Dióxido de azufre libre</td><td>mg/dm³</td></tr>
<tr><td>total_sulfur_dioxide</td><td>Dióxido de azufre libre y ligado</td><td>mg/dm³</td></tr>
<tr><td>density</td><td>Densidad del vino</td><td>g/cm³</td></tr>
<tr><td>pH</td><td>Acidez en escala logarítmica</td><td>Escala de pH</td></tr>
<tr><td>sulphates</td><td>Concentración de sulfatos</td><td>g/dm³</td></tr>
<tr><td>alcohol</td><td>Grado alcohólico</td><td>% vol.</td></tr></table>

<h2 id="reglas">2. Restricción de métodos: ¿por qué no usar cualquier modelo?</h2>
<p>La sección 2 de la guía evalúa los fundamentos del curso y limita el flujo a distancias, k-NN, k-means, PCA, SVM y MLP. La implementación respeta esa lista. Aunque aparezcan en las clases, Jaccard/Hamming, regresión lineal y CNN no forman parte de la solución de este reto: tenemos once mediciones continuas y una salida binaria, y la lista específica de la guía determina los métodos de entrega.</p>
<table><tr><th>Método</th><th>Qué hace aquí</th><th>Qué se puede escoger</th></tr>
<tr><td>Distancias y k-NN</td><td>Busca vinos cercanos y vota con sus etiquetas</td><td>Euclidiana, Manhattan, Minkowski y coseno; k; pesos</td></tr>
<tr><td>k-means</td><td>Descubre centroides y les asigna clase mayoritaria</td><td>Número de clústeres</td></tr>
<tr><td>PCA</td><td>Cambia la representación conservando varianza</td><td>Sin PCA, 90% o 95%; clasificador posterior</td></tr>
<tr><td>SVM</td><td>Busca una frontera con margen y penalización de errores</td><td>Lineal, polinomial, RBF; C, gamma, pesos de clase</td></tr>
<tr><td>MLP</td><td>Aprende representaciones con capas no lineales</td><td>Capas, ReLU/tanh, alpha y ponderación de clases</td></tr></table>

<h2 id="pasos">3. Paso a paso: el orden oficial de la guía</h2>
<p>Se sigue la secuencia de la página 3. Los intervalos siguientes son la distribución sugerida de los 120 minutos del hackathon, no una afirmación sobre cuánto trabajó cada integrante.</p>
<h3>Paso 1 · Configuración y roles (0–10 minutos)</h3>
<p><strong>Qué hicimos:</strong> organizamos carpetas, copiamos los CSV sin modificar sus originales y definimos un contrato común de columnas. La semilla es 42. La lógica central vive en <code>src/</code> y la reutilizan el notebook y Streamlit.</p>
<p><strong>Por qué:</strong> evita que cada módulo trate las variables de manera distinta. Los roles propuestos son datos y fundamentos; modelos y validación; aplicación y defensa. El equipo debe reflejar su distribución real de responsabilidades en sus contribuciones, sin atribuir a nadie tareas que no realizó.</p>
<h3>Paso 2 · Exploración y limpieza (10–30 minutos)</h3>
<p><strong>Qué hicimos:</strong> revisamos nulos, estadísticas, distribuciones, extremos IQR, correlaciones y perfiles repetidos. No se encontraron faltantes en las once variables de train ni test. Hay {audit['duplicate_feature_rows_train']} filas repetidas por sus mediciones en train y {audit['profiles_with_conflicting_labels']} perfil con etiquetas contradictorias. El identificador se conserva solo para alinear archivos.</p>
<p><strong>Por qué escalamos:</strong> el dióxido de azufre total llega a 440, mientras la densidad ronda 0,99. Sin escalado, la variable con mayor magnitud dominaría las distancias y alteraría los gradientes. StandardScaler expresa cada coordenada en desviaciones estándar.</p>
<div class="formula">z<sub>ij</sub> = (x<sub>ij</sub> − μ<sub>j</sub>) / σ<sub>j</sub><br>La media μ y la desviación σ se aprenden únicamente con la parte de entrenamiento de cada fold.</div>
<p><strong>Por qué no borramos todos los outliers:</strong> una bandera IQR no demuestra un error de laboratorio. Quitar puntos extremos podría eliminar ejemplos útiles de la clase minoritaria. Se conservan y se informa el diagnóstico.</p>
{corr_fig}{hist_fig}
<h3>Paso 3 · Baseline obligatorio (30–50 minutos)</h3>
<p><strong>Qué hicimos:</strong> comparamos k-NN con k=3, 5 y 7 y distancias Euclidiana y Manhattan sobre variables escaladas. Añadimos k-means con 2 y 16 clústeres. Los ocho baselines usan los mismos cinco folds estratificados: cada modelo se ajusta cinco veces, dejando fuera una quinta parte para validación en cada turno.</p>
<div class="formula">Minkowski: d<sub>p</sub>(x,z) = (Σ |x<sub>j</sub> − z<sub>j</sub>|<sup>p</sup>)<sup>1/p</sup><br>p=1 → Manhattan; p=2 → Euclidiana; se prueba también p=3.<br>Disimilitud coseno: 1 − (x·z)/(‖x‖ ‖z‖)</div>
<p>Coseno compara orientación respecto al centro de referencia después de estandarizar. No conserva la interpretación de proporciones de concentraciones crudas. Además, 1−coseno no cumple todos los axiomas de una métrica, como se explicó en Espacios y Métricas.</p>
<p>k-NN es supervisado y memoriza ejemplos. k-means encuentra centroides sin consultar etiquetas, minimizando la suma de distancias euclidianas cuadráticas. Para evaluar k-means como clasificador se aprende después una correspondencia clúster→clase mayoritaria usando exclusivamente las etiquetas del entrenamiento. Los números de clúster no son etiquetas.</p>
<div class="result">Mejor baseline: <strong>{escape(base['name'])}</strong>, con F1 macro <strong>{base.f1_macro_mean:.4f} ± {base.f1_macro_std:.4f}</strong>. Este resultado es la referencia que debe mejorar la progresión.</div>
<h3>Paso 4 · Feature engineering y PCA (50–75 minutos)</h3>
<p><strong>Qué hicimos:</strong> conservamos las once entradas y añadimos tres variables interpretables, superando el mínimo de dos de la guía:</p>
<table><tr><th>Cociente</th><th>Sentido de la hipótesis</th><th>Precaución</th></tr>
<tr><td>Acidez fija / volátil</td><td>Representa un balance relativo entre tipos de acidez</td><td>No es una medida completa de acidez ni de sabor</td></tr>
<tr><td>SO₂ libre / total</td><td>Representa la fracción de azufre que está libre</td><td>La interpretación presupone mediciones coherentes</td></tr>
<tr><td>Alcohol / densidad</td><td>Resume una relación empírica entre propiedades vinculadas</td><td>No es una ley de calidad ni un cociente adimensional</td></tr></table>
<p>Se calculan con las unidades originales, antes del escalado. Un denominador de magnitud menor o igual a 10⁻⁹ produce NaN; la mediana del entrenamiento del fold lo resuelve. También se prueban ocho distancias a centroides como variables geométricas; esos centroides se aprenden dentro del fold.</p>
<p><strong>PCA:</strong> cambia de ejes para concentrar la varianza en componentes ortogonales. Se prueban 90% y 95%. Para comprobar su efecto, una comparación mantiene fijos k=5 y Manhattan y cambia solo ratios/PCA.</p>
<div class="formula">Centrado/escalado → covarianza → autovectores ordenados por autovalor → proyección T=ZV.<br>Varianza explicada por componente j = λ<sub>j</sub> / Σλ.</div>
{pca_fig}{results_table(paired)}
<p>Estos datos permiten observar que una transformación nueva no garantiza aumentar F1. PCA prioriza varianza, no la etiqueta. Sus dos ejes dibujados tampoco son todos los componentes que usa un pipeline con retención del 95%. La búsqueda final eligió un candidato sin PCA.</p>
<h3>Paso 5 · SVM y MLP (75–100 minutos)</h3>
<p><strong>SVM:</strong> se evalúan kernels lineal, polinomial y RBF. Un C grande penaliza más las violaciones del margen; gamma controla cuánto decae la influencia en RBF. Se comparan C, gamma y pesos de clase. El polinomial de la búsqueda usa grado 3; el simulador permite explorar grados 2–4.</p>
<div class="formula">SVM: minimizar ½‖w‖² + CΣξ<sub>i</sub>, con restricciones de margen blando.<br>Kernel RBF: K(x,z)=exp(−γ‖x−z‖²).<br>Kernel polinomial: K(x,z)=(γx·z+r)<sup>d</sup>.</div>
<p><strong>MLP:</strong> se comparan (32,) y (64,32), activación ReLU/tanh y alpha de 0,0001 o 0,01. También se prueba ponderar clases en la pérdida. La salida binaria es sigmoide y el entrenamiento usa entropía cruzada regularizada y Adam. Backpropagation calcula el gradiente; el optimizador actualiza los pesos. Es la misma separación entre forward, pérdida, backward y actualización presentada en las clases de canciones.</p>
<div class="formula">h=φ(W₁x+b₁), p̂=σ(W₂h+b₂)<br>L=−[y ln(p̂)+(1−y)ln(1−p̂)]<br>θ ← θ − η∇L (forma básica de descenso; Adam adapta los pasos)</div>
<p>El límite es de 500 iteraciones y se registran avisos de convergencia. En esta ejecución hubo {warnings_count} avisos en total en la CV; el detalle está en <code>outputs/cv_folds.csv</code>. Llegar al límite no implica por sí solo convergencia. No se ocultan esos avisos ni se afirma que la pérdida de entrenamiento garantice generalización.</p>
<h3>Cómo evitamos fuga de datos</h3>
<div class="note"><strong>Flujo completo por fold:</strong> ratios opcionales → imputación → escalado → centroides y nuevo escalado opcionales → PCA opcional → clasificador. Se ajusta todo con la parte de entrenamiento y se transforma la validación sin volver a aprender sus estadísticas. Las etiquetas de test nunca intervienen.</div>
<p>Los gráficos descriptivos pueden usar todo train, pero sus objetos de PCA/escalado no se reutilizan para puntuar modelos. Los pesos balanceados de la MLP y el mapeo de clases de k-means también se calculan con el entrenamiento del fold.</p>
<h3>Paso 6 · Selección y entrega (100–120 minutos)</h3>
<p>Se elige el mayor F1 macro medio; los empates se resuelven por menor desviación y luego nombre. Se vuelve a entrenar el ganador con todo train y se generan las 1.300 predicciones en el orden del archivo de prueba. Estos son los principales resultados:</p>
{results_table(representative)}
<div class="result">Ganador: <strong>{escape(meta['best']['name'])}</strong>. Mejora frente al mejor baseline: <strong>{gain:.4f} puntos de F1 macro</strong> (escala 0–1). F1 de la clase superior: <strong>{winner.f1_1_mean:.4f}</strong>; recall superior: <strong>{winner.recall_1_mean:.4f}</strong>.</div>
<p>La votación ponderada da mayor influencia a vecinos cercanos. En coincidencias exactas, scikit-learn da peso a los vecinos de distancia cero. Con perfiles repetidos, esa propiedad puede ayudar en CV convencional. No se elige el modelo por ser más reciente o complejo: se elige con la métrica acordada.</p>
{confusion_fig}
<h3>Lo que los números permiten afirmar y lo que queda por medir</h3>
<p>La CV de selección dio {winner.f1_macro_mean:.4f}. Al agrupar perfiles idénticos para que no se compartan entre entrenamiento y validación, el diagnóstico del mismo ganador dio <strong>{grouped:.4f} ± {meta['grouped_diagnostic']['f1_macro_std']:.4f}</strong>. La reducción muestra sensibilidad a cómo se define una muestra nueva. Hay {audit['test_rows_with_profile_in_train']} filas de test cuyo perfil aparece en train; ese conteo es descriptivo y no aporta etiquetas ocultas.</p>
<p>Se han comparado muchas configuraciones con los mismos folds: el máximo puede ser optimista. La desviación entre folds describe variación, no un intervalo de confianza. La validación agrupada se hace después de elegir el ganador y no es anidada. La evaluación definitiva corresponde al test de la profesora; no conocemos su F1 ni el ranking.</p>

<h2 id="entregables">4. Entregables obligatorios y ubicación</h2>
<table><tr><th>Guía, sección 4</th><th>Archivo</th><th>Cómo se comprueba</th></tr>
<tr><td>Notebook limpio, comentado y ejecutado</td><td>notebooks/01_eda_y_modelado.ipynb</td><td>EDA, baseline reejecutado, PCA, resultados completos y controles SVM/MLP; celdas con salidas reales</td></tr>
<tr><td>Streamlit funcional</td><td>app/app.py y app/modules/</td><td>Seis módulos, figuras EDA/PCA, tabla de modelos y predicción interactiva; sin guion ni documentos internos</td></tr>
<tr><td>Predicciones</td><td>entregas/submission.csv</td><td>Exactamente id,prediction; 1.300 registros más cabecera, sin nulos y con orden conservado</td></tr>
<tr><td>Resumen y tabla de CV</td><td>README.md</td><td>Instalación, enfoque matemático, resultados medidos y límites</td></tr>
<tr><td>Presentación para clase</td><td>presentacion/PRESENTACION_CLASE.html</td><td>Ocho diapositivas independientes, navegación con flechas, pantalla completa e impresión a PDF</td></tr>
<tr><td>Preparación interna del equipo</td><td>docs/grupo/EXPLICACION.html y docs/grupo/GUION_EXPOSICION.html</td><td>Justificaciones, preguntas de defensa y guion; no se muestran dentro de la app</td></tr>
<tr><td>Continuidad para asistentes IA</td><td>CONTEXT.md y AGENTS.md</td><td>Fuentes, reglas, arquitectura y comandos para continuar sin perder contexto</td></tr></table>
<p>El repositorio de destino es <a href="{REPO}">Hackaton-Grupo-1</a>. El historial debe registrar las contribuciones reales. Los roles y commits de los demás integrantes no se simulan. El archivo CSV se valida automáticamente antes de exportarse.</p>

<h2 id="rubrica">5. Rúbrica: dónde se atiende cada criterio</h2>
<table><tr><th>Criterio</th><th>Peso</th><th>Evidencia y límite</th></tr>
<tr><td>EDA, preprocesamiento y variables</td><td>20%</td><td>Gráficas, auditoría, justificación del escalado, tres cocientes y pipeline por fold</td></tr>
<tr><td>Progresión de modelos</td><td>25%</td><td>Baselines, ablation ratios/PCA, SVM y MLP; F1 y tiempos con folds comunes</td></tr>
<tr><td>Desempeño oculto</td><td>25%</td><td>submission válido; puntaje y ranking dependen de la evaluación docente</td></tr>
<tr><td>App y orden en GitHub</td><td>15%</td><td>Módulos y estructura solicitada; colaboración real debe constar en GitHub</td></tr>
<tr><td>Pitch y defensa</td><td>15%</td><td>Guion en docs/grupo/GUION_EXPOSICION.html; la exposición y preguntas corresponden al equipo</td></tr></table>

<h2 id="uso">Cómo usar la aplicación, sin leer el código</h2>
<ol><li>Abre <code>iniciar_app.bat</code> en Windows o ejecuta el comando Streamlit del README.</li><li>Recorre «Módulo» de 1 a 6 para explorar los datos y resultados.</li><li>Para ir directo a un algoritmo, usa «¿Qué método quieres usar?».</li><li>Escoge una configuración ya evaluada. F1 macro y su desviación corresponden a sus cinco folds.</li><li>Si quieres experimentar, activa «Ajustar parámetros del método». Evaluar cinco folds vuelve a entrenar y puede tardar; no es necesario para mostrar predicciones.</li><li>Introduce las once mediciones y pulsa «Predecir calidad». La MLP produce una estimación no calibrada; k-NN muestra votos, k-means proporción del clúster y SVM un puntaje de decisión.</li><li>En el módulo 6 descarga el CSV de la entrega. El CSV corresponde al ganador reproducible, no a una prueba manual aislada.</li></ol>
<h3>¿Cómo evitamos entrenar durante la exposición?</h3>
<p>Los archivos <code>.joblib</code> guardan el pipeline completo: ratios, imputación, escalado, PCA o centroides cuando corresponden y el clasificador aprendido. <code>scripts/preparar_demo.py</code> prepara las 79 configuraciones evaluadas una vez. La aplicación las carga de disco y las conserva además en memoria con caché.</p>
<p>Cerrar y volver a abrir la app no obliga a entrenarlas de nuevo. Si se cambian parámetros a una configuración inédita, el primer ajuste puede tardar; después también queda guardado. Cambiar los datos o la versión de scikit-learn invalida esos archivos y requiere prepararlos de nuevo. Para presentar, conviene elegir configuraciones evaluadas y evitar el botón de evaluación de cinco folds.</p>
<div class="note">La comparación futura entre tres proyectos todavía no se ha realizado. La carpeta <code>Hackathon_MMMIA_Grupo1 (1)</code> se conserva sin analizar ni integrar; los resultados de este documento pertenecen únicamente al proyecto actual.</div>

<h2 id="fuentes">Conexión con las clases del semestre</h2>
<p>Se revisaron los 13 materiales de clase y guía proporcionados. Los PDF de SVM y geometría neuronal, así como las páginas gráficas de Deep Learning, se revisaron visualmente. No se abrieron los enlaces externos de Colab/Drive: estas referencias se limitan a los archivos locales.</p>
<table><tr><th>Fuente local</th><th>Parte revisada y concepto</th><th>Aplicación en el trabajo</th></tr>
<tr><td>EspaciosYMetricas.pdf</td><td>pp. 9–19: espacio métrico y L2; 20–28: L1/Lp; 31–32: coseno</td><td>Selector de distancias y aclaración matemática de coseno</td></tr>
<tr><td>20260806_MMMIA_SolanoCorreaYadyTatiana_Sesión2.pdf</td><td>pp. 15–22: k-NN y escalado; 23–33: k-means y Lloyd</td><td>Baseline, vecinos, centroides y z-score</td></tr>
<tr><td>Unidad3_Bloque1.html</td><td>Límites de distancias, dimensionalidad, inferencia y celdas de Voronoi</td><td>Comparar tiempos y no atribuir geometrías arbitrarias a k-means</td></tr>
<tr><td>Unidad3_Bloque2.html</td><td>Embeddings, XOR, expansión y kernel trick</td><td>Justificación de variables nuevas y kernels no lineales</td></tr>
<tr><td>Unidad3_Bloque3.html</td><td>Normas, producto punto, hiperplano, distancia y margen</td><td>Fundamento de SVM y significado de su puntaje</td></tr>
<tr><td>Unidad3_Bloque4.html</td><td>Clasificación vs. regresión; promedio local y OLS</td><td>Explicar por qué aquí se predicen clases, no valores continuos</td></tr>
<tr><td>PCA_TeoríaConEjemplo.html</td><td>Centrado, covarianza, autovalores, autovectores y proyección</td><td>PCA de 90/95% y reproducción del ejemplo de cuatro estudiantes en notebook</td></tr>
<tr><td>1.Máquinas_de_Soporte_Vectorial.pdf</td><td>pp. 3–10: hiperplano/margen/C; 13–14: kernels/gamma; 15–20: métricas y pipelines</td><td>SVM lineal/poly/RBF y ajuste sin fuga</td></tr>
<tr><td>20261029_1_Intro_ANN_MLP.pdf</td><td>pp. 10–13: neurona; 19–23: MLP; 24–32: pérdida, gradiente y backpropagation</td><td>MLP, ReLU/tanh, entropía cruzada y distinción aprendizaje/inferencia</td></tr>
<tr><td>20261029_2_Geometría_del_Aprendizaje_Neuronal.pdf</td><td>pp. 4–14: pesos, sesgo, sigmoide, error y evolución geométrica</td><td>Interpretación del cambio de frontera y de la pérdida de entrenamiento</td></tr>
<tr><td>20261029_3_EjercicioMatemáticoANNCanciones.pdf</td><td>pp. 3–11: forward, regla de la cadena, gradientes y actualización</td><td>Explicación de cómo aprende la red; no tratar backpropagation como optimizador</td></tr>
<tr><td>20261112_DelMLPalDL.pdf</td><td>pp. 2–12: grafo, convolución, filtros y pooling; 17: recorrido geométrico</td><td>Ubicar MLP en el curso y justificar que las CNN quedan fuera de este reto tabular</td></tr>
<tr><td>Guia_Estudiantes_Hackathon_IA.pdf</td><td>pp. 1–2: datos; 3: métodos y seis pasos; 4–5: entregables y rúbrica</td><td>Orden del notebook/app, F1 macro, requisitos de CSV y defensa</td></tr></table>

<h2 id="preguntas">Preguntas que debemos poder responder</h2>
<h3>¿Por qué ganó k-NN y no una red?</h3><p>Porque el criterio de selección fue F1 macro medido y k-NN obtuvo el mayor promedio en la rejilla definida. El resultado depende de representación, muestra y partición; no establece que k-NN sea universalmente superior. Los perfiles repetidos ayudan a explicar por qué hacemos un control agrupado.</p>
<h3>¿Los tres ratios mejoraron por sí solos?</h3><p>La ablación k=5/Manhattan muestra que no hay una mejora automática. El ganador cambia además distancia y pesos; su mejora conjunta no se atribuye exclusivamente a los cocientes.</p>
<h3>¿Por qué no usar siempre class_weight=balanced?</h3><p>Aumenta el costo relativo de equivocarse con la minoría, pero puede producir más falsos positivos. Se comparan ambas opciones y se decide con F1 macro. Balancear la pérdida no garantiza mejorar la métrica.</p>
<h3>¿El puntaje SVM es confianza porcentual?</h3><p>No. Es una función de decisión con umbral 0. En SVC no se muestra como distancia euclidiana normalizada ni se convierte a porcentaje aplicando sigmoide sin calibración.</p>
<h3>¿Ya sabemos que superamos a la profesora?</h3><p>No. Tenemos un CSV válido y un resultado de validación. El baseline docente, las etiquetas del test y el ranking no están disponibles. Solo la evaluación oficial resuelve esa parte de la rúbrica.</p>
"""
    (ROOT / "docs/grupo/EXPLICACION.html").write_text(shell("Grupo 1 · Explicación del hackathon", body), encoding="utf-8")

    pitch = f"""<div class="eyebrow">GRUPO 1 · GUION INTERNO DE EXPOSICIÓN</div><h1>Cinco minutos para explicar el trabajo</h1>
<p class="lead">Apoyo para hablar con sus propias palabras. Distribuyan las intervenciones según el trabajo real del equipo.</p>
<div class="note">Este guion es para el equipo. Proyectar <a href="../../presentacion/PRESENTACION_CLASE.html">las ocho diapositivas de clase</a>; la demostración de la aplicación se abre por separado. El guion no aparece en la app ni en las diapositivas.</div>
<p class="print"><button onclick="window.print()">Imprimir o guardar como PDF</button></p>
<h2>0:00–0:40 · Problema y métrica</h2><p>«Clasificamos vinos estándar y superiores usando once mediciones. Aprendemos con 5.197 muestras y predecimos 1.300. Como solo el 15,93% del entrenamiento es superior, usamos F1 macro: la clase minoritaria cuenta tanto como la mayoritaria».</p>
<h2>0:40–1:30 · EDA y representación</h2><p>«No encontramos nulos. Las escalas son muy distintas: densidad cerca de uno frente a SO₂ total hasta 440. Por eso escalamos dentro de cada fold. Conservamos valores extremos sin evidencia de error y añadimos tres cocientes: acidez fija/volátil, SO₂ libre/total y alcohol/densidad».</p>
<h2>1:30–2:20 · Baseline y PCA</h2><p>«Comparamos k-NN con 3, 5 y 7 vecinos y Euclidiana/Manhattan; también k-means por mayoría. El mejor baseline dio {base.f1_macro_mean:.4f}. Luego probamos ratios, distancias a centroides y PCA de 90% y 95%. Una ablación mantuvo k y distancia fijos para aislar el cambio de representación».</p>
<h2>2:20–3:20 · Progresión a SVM y MLP</h2><p>«SVM compara tres kernels y ajusta C, gamma y pesos de clase. MLP compara una o dos capas, ReLU/tanh y regularización. Todos usan cinco folds iguales, semilla 42 y pipeline completo. No asumimos que el modelo más complejo vaya a ganar».</p>
<h2>3:20–4:20 · Resultado y límite</h2><p>«El ganador fue k-NN con 11 vecinos, coseno y votos ponderados: F1 macro {winner.f1_macro_mean:.4f} con desviación {winner.f1_macro_std:.4f}; F1 superior {winner.f1_1_mean:.4f}. Detectamos 807 filas repetidas por perfil. Al agrupar perfiles idénticos, el control dio {grouped:.4f}. La diferencia nos obliga a distinguir nuevas filas de nuevos perfiles. Esta CV sirve para selección; no sustituye el test oculto».</p>
<h2>4:20–5:00 · Demostración y entrega</h2><p>«Streamlit recorre los seis pasos de la guía y permite elegir cada método, cambiar parámetros e introducir once mediciones. La entrega contiene notebook ejecutado, README, documentación y CSV comprobado: dos columnas y 1.300 observaciones en el orden original. El puntaje oculto lo calcula la profesora».</p>
<h2>Preguntas rápidas</h2><table><tr><th>Pregunta</th><th>Respuesta clave</th></tr>
<tr><td>¿Fuga de datos?</td><td>Imputación, escalado, centroides y PCA se ajustan dentro del entrenamiento de cada fold.</td></tr>
<tr><td>¿Qué hace C?</td><td>Regula cuánto se penalizan las violaciones del margen de SVM.</td></tr>
<tr><td>¿Qué hace gamma?</td><td>En RBF controla la velocidad con que decae la similitud con la distancia.</td></tr>
<tr><td>¿Qué hace alpha?</td><td>Regulariza los pesos de la MLP; no es la tasa de aprendizaje.</td></tr>
<tr><td>¿Por qué no PCA final?</td><td>Se probó, pero el máximo F1 de la rejilla correspondió a un candidato sin PCA.</td></tr>
<tr><td>¿Coseno es métrica?</td><td>1−coseno es disimilitud y no satisface todos los axiomas métricos.</td></tr>
<tr><td>¿k-means ya clasifica?</td><td>Agrupa sin etiquetas; el mapeo por mayoría se aprende después, solo con train del fold.</td></tr>
<tr><td>¿Hay convergencia garantizada?</td><td>No. Se conservan los avisos de ajuste y se informa el límite de iteraciones.</td></tr>
<tr><td>¿La CV es imparcial después de elegir?</td><td>Puede ser optimista al seleccionar entre 79 candidatos; el test oculto queda separado.</td></tr></table>
<p><a href="EXPLICACION.html">Abrir la explicación completa</a> · <a href="{REPO}">Repositorio</a></p>"""
    (ROOT / "docs/grupo/GUION_EXPOSICION.html").write_text(shell("Grupo 1 · Pitch de cinco minutos", pitch), encoding="utf-8")

    table_md = "| Configuración | F1 macro | Desv. folds | F1 superior | Ajuste s/fold | Predicción s/fold |\n|---|---:|---:|---:|---:|---:|\n"
    for _, row in representative.iterrows():
        table_md += f"| `{row['name']}` | {row.f1_macro_mean:.4f} | {row.f1_macro_std:.4f} | {row.f1_1_mean:.4f} | {row.fit_seconds_mean:.4f} | {row.prediction_seconds_mean:.4f} |\n"
    readme = f"""# Grupo 1 · Hackathon de Métodos y Modelos Matemáticos para IA

**Juan Alejandro Cardona · Javier Urrego · Santiago Osorio**

Profesora: Tatiana Solano · Pontificia Universidad Javeriana Cali

Repositorio: [{REPO}]({REPO})

Clasificamos vinos de **calidad estándar (0)** o **superior (1)** con once mediciones
fisicoquímicas. El proyecto sigue los **seis pasos de la guía oficial** y usa únicamente
los métodos permitidos del semestre. Incluye notebook ejecutado, aplicación modular,
comparación reproducible, predicciones y una explicación de cada decisión.

El ganador de {meta['candidates']} configuraciones fue **k-NN, k=11, coseno, votos ponderados por
distancia y tres cocientes**, con **F1 macro CV {winner.f1_macro_mean:.4f} ± {winner.f1_macro_std:.4f}**.
El control agrupando perfiles idénticos dio **{grouped:.4f}**. El **F1 del test oculto no se conoce**.

## Empezar aquí

- **Para mostrar en clase:** [presentación de ocho diapositivas](presentacion/PRESENTACION_CLASE.html)
  y aplicación VinoLab. Abre `abrir_presentacion.bat` para las diapositivas e
  `iniciar_app.bat` para la app; son independientes.
- **Para preparar al grupo:** carpeta `docs/grupo/`. Abre `abrir_material_grupo.bat`
  para estudiar la explicación. El guion y las respuestas de defensa quedan fuera de la app.
- [Documento explicado paso a paso](docs/grupo/EXPLICACION.html): qué se hizo, por qué,
  resultados, rúbrica y conexión con todas las clases. Abrir en navegador; funciona
  sin internet y tiene botón para imprimir o guardar como PDF.
- [Notebook ejecutado](notebooks/01_eda_y_modelado.ipynb): código, gráficas y salidas reales.
- [Guion interno de exposición de 5 minutos](docs/grupo/GUION_EXPOSICION.html).
- [Contexto para otras IA](CONTEXT.md) y [reglas del repositorio](AGENTS.md).
- [CSV de entrega](entregas/submission.csv).

## Ejecutar la aplicación

En este equipo ya se creó `.venv` y se instalaron las dependencias. Basta abrir
**`iniciar_app.bat`** o ejecutar desde esta carpeta:

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app/app.py
```

Abrir **http://localhost:8501**. Detener el servidor con `Ctrl+C`.

Para instalar desde cero, usar **Python 3.13** (probado con {meta['versions']['python']}):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m streamlit run app/app.py
```

No hace falta activar el entorno ni cambiar la política de ejecución de PowerShell.
Los datos y resultados completos vienen incluidos. Los modelos se entrenan localmente;
la aplicación no necesita enviar los CSV a un servicio externo.

### Navegación y métodos

La barra lateral contiene «Módulo» y «¿Qué método quieres usar?». Mantiene el orden
de la guía para los contenidos técnicos, con estos módulos:

1. **Resumen del proyecto**: problema, datos y métodos disponibles.
2. **EDA y limpieza**: desbalance, correlaciones, distribuciones y extremos IQR.
3. **Baseline por distancias**: k-NN y k-means; configuraciones, métricas y simulador.
4. **Variables nuevas y PCA**: tres ratios, comparación controlada y proyecciones.
5. **SVM y MLP**: SVM lineal/polinomial/RBF y redes, con parámetros ajustables.
6. **Resultados y predicciones**: tabla completa, diagnóstico agrupado, CSV y simulador ganador.

La app no contiene guiones, reparto de roles ni descargas del material interno.
Las diapositivas tienen flechas, pantalla completa (F o F11) y un botón para imprimir
o guardar como PDF. Funcionan sin internet y no requieren ejecutar Streamlit.

Puedes escoger cualquier método directamente. k-NN permite Euclidiana, Manhattan,
Minkowski y coseno. En cada panel se puede elegir una configuración evaluada o activar
«Ajustar parámetros del método». La evaluación personalizada no sobrescribe los
resultados completos ni cambia `entregas/submission.csv`.

El simulador recibe **once mediciones**, no id ni target. SVM muestra su función de
decisión con umbral 0; k-NN muestra votos ponderados; k-means proporción en el clúster;
MLP una estimación de salida. Ninguna se presenta como probabilidad calibrada.

### Modelos guardados y rapidez durante la presentación

Sí: los modelos pueden guardarse. Inicialmente se guardaron seis representantes en
`outputs/model_*.joblib`. Además, se prepararon las **79 configuraciones evaluadas**
en `outputs/demo_models/`, con índice en `manifest.json`. Cada archivo contiene el
pipeline completo y se comprueban sus parámetros, el hash de entrenamiento y la
versión de scikit-learn antes de reutilizarlo.

La app carga estos archivos y usa caché de memoria: **seleccionar una configuración
evaluada y predecir no vuelve a entrenar**. Los ajustes personalizados nuevos sí
necesitan una primera preparación, que después se guarda. El botón «Evaluar esta
configuración con 5 folds» vuelve a entrenar y conviene usarlo antes de exponer.

Para preparar la demo en otro equipo o tras cambiar datos/versiones:

```powershell
.\\.venv\\Scripts\\python.exe scripts/preparar_demo.py
```

Esto no repite la búsqueda cruzada, no cambia las métricas y no sustituye el CSV final.
Cerrar la app no borra los archivos de modelos. Mantén `src/`, los datos y las
versiones compatibles junto con esos archivos al trasladar el proyecto.

## Datos y comprobaciones

| Archivo | Registros | Columnas | Uso |
|---|---:|---:|---|
| `data/train_public.csv` | 5.197 | 13 | id + 11 variables + target |
| `data/test_public_features.csv` | 1.300 | 12 | id + 11 variables, sin etiquetas |
| `entregas/submission.csv` | 1.300 | 2 | id,prediction |

`train_public.csv` es una copia exacta de `train_public (1).csv`; los originales en
`../Clases` se conservan. Hay **4.369 ceros y 828 unos**, sin faltantes en los predictores.
`id` se excluye de todos los modelos.

Se identificaron **807 filas repetidas por perfil** en train, un perfil con etiquetas
contradictorias y 337 filas de test con perfiles presentes en train. No se borran
automáticamente ni se usan identificadores para memorizar etiquetas. Se añade un
control agrupado y se informa el límite de generalización.

El CSV final tiene exactamente `id,prediction`, clases 0/1, ningún nulo y conserva el
orden original de test. Son **1.300 observaciones + 1 cabecera = 1.301 líneas**.

## Enfoque matemático y prevención de fuga

Cada muestra pertenece a R¹¹. Las escalas difieren mucho: densidad cerca de 0,99 frente
a SO₂ total hasta 440. Se aplica z-score para que la magnitud de una variable no domine
las distancias y para facilitar la optimización de las redes.

Se conservan las once entradas y se crean tres cocientes antes del escalado:

- `fixed_acidity / volatile_acidity`: balance relativo entre acidez fija y volátil.
- `free_sulfur_dioxide / total_sulfur_dioxide`: fracción de SO₂ libre.
- `alcohol / density`: relación empírica entre propiedades relacionadas.

Son hipótesis de representación, no leyes causales de calidad. Denominadores casi
nulos (≤10⁻⁹) producen NaN y se imputan por mediana. También se prueban distancias a
ocho centroides. El pipeline se ajusta **dentro de cada fold**:

```text
11 variables → ratios opcionales → mediana → StandardScaler
             → centroides y nuevo escalado opcionales → PCA opcional → clasificador
```

La mediana, media, desviación, centroides y componentes nunca se aprenden con las filas
de validación. El PCA descriptivo de las figuras se mantiene separado. k-means aprende
su mapeo clúster→clase solo con el entrenamiento. La MLP calcula pesos de clase solo
con ese fold y usa entropía cruzada regularizada y Adam; no optimiza F1 directamente.

## Resultados de validación cruzada

**{meta['candidates']} configuraciones × 5 folds**, `StratifiedKFold(shuffle=True, random_state=42)`.
Misma partición para todos. Métrica de selección: promedio del F1 macro por fold;
empate por menor desviación y luego nombre. Tiempos medidos en este equipo, con un
hilo por ajuste; no representan benchmarks universales.

{table_md}
Tabla completa: [outputs/cv_results.csv](outputs/cv_results.csv). Evidencia por fold:
[outputs/cv_folds.csv](outputs/cv_folds.csv). Particiones:
[outputs/fold_assignments.csv](outputs/fold_assignments.csv).

La mejora del ganador frente al mejor baseline es **{gain:.4f}** en escala 0–1.
F1 superior **{winner.f1_1_mean:.4f}**, precisión superior **{winner.precision_1_mean:.4f}**,
recall superior **{winner.recall_1_mean:.4f}**. La matriz OOF es `{meta['oof_confusion_matrix']}`
(filas reales, columnas predichas, orden 0/1). El ganador no usa PCA. La ablación
k=5/Manhattan del notebook muestra que añadir ratios no mejora automáticamente el
resultado; no atribuimos la mejora total solo a esas variables.

### Alcance de los resultados

- La desviación entre folds **no es un intervalo de confianza**.
- Elegir el máximo de 79 configuraciones con la misma CV puede dar optimismo.
- El control `StratifiedGroupKFold` del ganador da **{grouped:.4f} ± {meta['grouped_diagnostic']['f1_macro_std']:.4f}**;
  agrupa perfiles idénticos, pero se hizo tras seleccionar el ganador. No es CV anidada
  ni un conjunto independiente de evaluación.
- Los {warnings_count} avisos de ajuste en CV están conservados en `outputs/cv_folds.csv`.
  Un aviso de MLP al alcanzar 500 iteraciones no se oculta ni se trata como convergencia.
- No conocemos el baseline docente, el ranking ni las etiquetas ocultas. No se afirma
  haber superado al docente o conseguido un F1 de leaderboard.

## Reproducir entrenamiento, documentos y notebook

```powershell
.\\.venv\\Scripts\\python.exe scripts/entrenar.py --profile complete
.\\.venv\\Scripts\\python.exe scripts/crear_documentacion.py
.\\.venv\\Scripts\\python.exe scripts/crear_presentacion.py
.\\.venv\\Scripts\\python.exe scripts/preparar_demo.py
.\\.venv\\Scripts\\python.exe scripts/crear_notebook.py --execute
.\\.venv\\Scripts\\python.exe -m pytest tests -q
.\\.venv\\Scripts\\python.exe scripts/verificar_entrega.py
```

La búsqueda completa tardó aproximadamente **{meta['elapsed_seconds']/60:.1f} minutos** en esta ejecución;
el tiempo depende del equipo. `--profile quick` sirve para ensayos y escribe en
`outputs/quick/`, sin sustituir la entrega completa. Las salidas originales completas
vienen incluidas para abrir la app sin repetir la búsqueda.

El notebook vuelve a ejecutar los baselines, la ablación y representantes SVM/MLP;
lee la búsqueda completa desde archivos medidos. Cambiar `REENTRENAR_TODO=True`
repite toda la rejilla. Si faltan resultados, la genera automáticamente. Las celdas
contienen sus contadores y salidas; no se fabrican resultados de notebook.

Versiones principales: Python {meta['versions']['python']}, scikit-learn {meta['versions']['sklearn']},
NumPy {meta['versions']['numpy']} y pandas {meta['versions']['pandas']}. Dependencias directas fijadas
en `requirements.txt` y entorno completo registrado en `requirements.lock.txt`.
Los binarios `.joblib` son propios del proyecto y requieren versiones compatibles;
si se cambia de entorno y falla su carga, regenerarlos con el comando de entrenamiento.

### Qué se verificó

Pruebas automatizadas de contratos, fuga de datos, cobertura de la rejilla, los seis
módulos de Streamlit y predicción con los siete accesos directos. Se comprueban
además el notebook ejecutado con imágenes, los hashes de los datos y la alineación del
CSV. La conexión de inspección visual en navegador no estuvo disponible en esta sesión;
la navegación se verificó con el motor de pruebas de Streamlit.

## Estructura

```text
app/app.py                      Entrada Streamlit
app/common.py                   Selección, parámetros, métricas y simulador
app/modules/                    Un módulo por paso de la guía
data/                           CSV originales copiados
presentacion/PRESENTACION_CLASE.html Diapositivas para mostrar a la clase
docs/grupo/EXPLICACION.html            Documento autónomo y trazabilidad con clases
docs/grupo/GUION_EXPOSICION.html  Guion y preguntas para uso del equipo
entregas/submission.csv          Predicciones listas para evaluación docente
notebooks/01_eda_y_modelado.ipynb Notebook ejecutado
src/                            Datos, features, modelos, evaluación y gráficas
scripts/                        Entrenamiento, generación y verificación
outputs/                        Métricas, auditoría, figuras y modelos finales
outputs/demo_models/            79 pipelines preparados para la demo
tests/                          Fuga de datos, contratos y pruebas Streamlit
CONTEXT.md                      Contexto para otras IA
AGENTS.md                       Reglas de mantenimiento
```

## Relación con la guía y el semestre

El orden es el de la **página 3** de `Guia_Estudiantes_Hackathon_IA.pdf`; los entregables
siguen su **página 4**. El documento explicativo incluye una tabla con los 13 materiales,
sus páginas o bloques y cómo se utilizaron. SVM y geometría neuronal son PDF escaneados
y se revisaron visualmente. No se abrieron los Colab/Drive externos de las diapositivas.

Se utilizaron distancias, k-NN, k-means, PCA, SVM y MLP. Se excluyen árboles, Random
Forest, boosting y AutoML por la guía. Regresión, CNN y Jaccard/Hamming se explican
como contexto del semestre, pero quedan fuera de la lista de herramientas de este reto.

## Entrega en GitHub y responsabilidades

La carpeta añadida `Hackathon_MMMIA_Grupo1 (1)` no se modifica, analiza ni integra.
La comparación entre tres proyectos y la selección de sus mejores modelos quedan
para una etapa posterior. Los resultados actuales son solo de este proyecto.

Destino: [Hackaton-Grupo-1]({REPO}), rama `main`. Deben estar los cuatro archivos
obligatorios en sus rutas exactas. Los materiales de la profesora no se redistribuyen
en el repositorio: se incluyen las referencias y las copias de los CSV de trabajo.

El historial colaborativo y la exposición deben corresponder a contribuciones reales
de Juan Alejandro Cardona, Javier Urrego y Santiago Osorio. No se generan commits de
otros integrantes ni se asignan responsabilidades ficticias. El desempeño oculto y la
defensa oral son partes de la rúbrica que debe evaluar la profesora.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("README.md, docs/grupo/EXPLICACION.html y docs/grupo/GUION_EXPOSICION.html actualizados con resultados reales.")


if __name__ == "__main__":
    main()
