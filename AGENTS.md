# Guía para trabajar en este repositorio

Leer primero `CONTEXT.md`, `README.md` y `outputs/experiment.json` si existe.

- Mantener el orden de seis pasos de la guía y la interfaz en español.
- Usar exclusivamente los métodos académicos permitidos descritos en CONTEXT.md.
- Compartir la lógica en `src/`; evitar copias divergentes en notebook o aplicación.
- Excluir id y target de los predictores y preservar el orden de test al exportar.
- Ajustar imputación, escalado, centroides y PCA dentro del fold de entrenamiento.
- No presentar CV de selección como evaluación independiente, ni inventar el F1 oculto.
- Registrar los avisos de convergencia; no ocultarlos con filtros globales.
- Si se modifica el entrenamiento, regenerar resultados, documentación y notebook.
- Ejecutar `.venv/Scripts/python.exe -m pytest tests -q` y
  `.venv/Scripts/python.exe scripts/verificar_entrega.py` antes de cerrar cambios importantes.
- No instalar dependencias globalmente. Usar `.venv` y mantener `requirements.txt`.
- No cambiar ni borrar los archivos originales de la carpeta hermana `Clases`.
- No generar contribuciones o commits ficticios para cumplir la rúbrica colaborativa.
