# Grupo 1 · Hackathon de Métodos y Modelos Matemáticos para IA

**Juan Alejandro Cardona · Javier Urrego · Santiago Osorio**

Profesora: Tatiana Solano · Pontificia Universidad Javeriana Cali

Repositorio: [https://github.com/Sant0s0/Hackaton-Grupo-1](https://github.com/Sant0s0/Hackaton-Grupo-1)

Clasificamos vinos de **calidad estándar (0)** o **superior (1)** con once mediciones
fisicoquímicas. El proyecto sigue los **seis pasos de la guía oficial** y usa únicamente
los métodos permitidos del semestre. Incluye notebook ejecutado, aplicación modular,
comparación reproducible, predicciones y una explicación de cada decisión.

El ganador de 79 configuraciones fue **k-NN, k=11, coseno, votos ponderados por
distancia y tres cocientes**, con **F1 macro CV 0.8510 ± 0.0175**.
El control agrupando perfiles idénticos dio **0.8041**. El **F1 del test oculto no se conoce**.

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
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Abrir **http://localhost:8501**. Detener el servidor con `Ctrl+C`.

Para instalar desde cero, usar **Python 3.13** (probado con 3.13.6):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/app.py
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
.\.venv\Scripts\python.exe scripts/preparar_demo.py
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

**79 configuraciones × 5 folds**, `StratifiedKFold(shuffle=True, random_state=42)`.
Misma partición para todos. Métrica de selección: promedio del F1 macro por fold;
empate por menor desviación y luego nombre. Tiempos medidos en este equipo, con un
hilo por ajuste; no representan benchmarks universales.

| Configuración | F1 macro | Desv. folds | F1 superior | Ajuste s/fold | Predicción s/fold |
|---|---:|---:|---:|---:|---:|
| `fe_knn_k11_cosine_distance` | 0.8510 | 0.0175 | 0.7478 | 0.0068 | 0.0804 |
| `mlp_64_32_relu_a0.01_bTrue` | 0.8235 | 0.0264 | 0.7076 | 4.9380 | 0.0015 |
| `svm_rbf_C10_g0.03_wNone` | 0.8138 | 0.0110 | 0.6796 | 0.2119 | 0.0755 |
| `svm_linear_C0.1_gscale_wNone` | 0.8089 | 0.0159 | 0.6732 | 0.1246 | 0.0158 |
| `base_knn_k3_manhattan` | 0.8066 | 0.0184 | 0.6731 | 0.0061 | 0.0400 |
| `svm_poly_C10_gscale_wNone` | 0.7923 | 0.0090 | 0.6414 | 0.6177 | 0.0171 |
| `base_kmeans_2` | 0.7665 | 0.0146 | 0.6288 | 0.0208 | 0.0018 |

Tabla completa: [outputs/cv_results.csv](outputs/cv_results.csv). Evidencia por fold:
[outputs/cv_folds.csv](outputs/cv_folds.csv). Particiones:
[outputs/fold_assignments.csv](outputs/fold_assignments.csv).

La mejora del ganador frente al mejor baseline es **0.0444** en escala 0–1.
F1 superior **0.7478**, precisión superior **0.7768**,
recall superior **0.7210**. La matriz OOF es `[[4197, 172], [231, 597]]`
(filas reales, columnas predichas, orden 0/1). El ganador no usa PCA. La ablación
k=5/Manhattan del notebook muestra que añadir ratios no mejora automáticamente el
resultado; no atribuimos la mejora total solo a esas variables.

### Alcance de los resultados

- La desviación entre folds **no es un intervalo de confianza**.
- Elegir el máximo de 79 configuraciones con la misma CV puede dar optimismo.
- El control `StratifiedGroupKFold` del ganador da **0.8041 ± 0.0121**;
  agrupa perfiles idénticos, pero se hizo tras seleccionar el ganador. No es CV anidada
  ni un conjunto independiente de evaluación.
- Los 90 avisos de ajuste en CV están conservados en `outputs/cv_folds.csv`.
  Un aviso de MLP al alcanzar 500 iteraciones no se oculta ni se trata como convergencia.
- No conocemos el baseline docente, el ranking ni las etiquetas ocultas. No se afirma
  haber superado al docente o conseguido un F1 de leaderboard.

## Reproducir entrenamiento, documentos y notebook

```powershell
.\.venv\Scripts\python.exe scripts/entrenar.py --profile complete
.\.venv\Scripts\python.exe scripts/crear_documentacion.py
.\.venv\Scripts\python.exe scripts/crear_presentacion.py
.\.venv\Scripts\python.exe scripts/preparar_demo.py
.\.venv\Scripts\python.exe scripts/crear_notebook.py --execute
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts/verificar_entrega.py
```

La búsqueda completa tardó aproximadamente **8.6 minutos** en esta ejecución;
el tiempo depende del equipo. `--profile quick` sirve para ensayos y escribe en
`outputs/quick/`, sin sustituir la entrega completa. Las salidas originales completas
vienen incluidas para abrir la app sin repetir la búsqueda.

El notebook vuelve a ejecutar los baselines, la ablación y representantes SVM/MLP;
lee la búsqueda completa desde archivos medidos. Cambiar `REENTRENAR_TODO=True`
repite toda la rejilla. Si faltan resultados, la genera automáticamente. Las celdas
contienen sus contadores y salidas; no se fabrican resultados de notebook.

Versiones principales: Python 3.13.6, scikit-learn 1.9.1,
NumPy 2.5.5 y pandas 3.1.1. Dependencias directas fijadas
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

Destino: [Hackaton-Grupo-1](https://github.com/Sant0s0/Hackaton-Grupo-1), rama `main`. Deben estar los cuatro archivos
obligatorios en sus rutas exactas. Los materiales de la profesora no se redistribuyen
en el repositorio: se incluyen las referencias y las copias de los CSV de trabajo.

El historial colaborativo y la exposición deben corresponder a contribuciones reales
de Juan Alejandro Cardona, Javier Urrego y Santiago Osorio. No se generan commits de
otros integrantes ni se asignan responsabilidades ficticias. El desempeño oculto y la
defensa oral son partes de la rúbrica que debe evaluar la profesora.
