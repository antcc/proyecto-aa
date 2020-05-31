- Eliminar TODO.md y NOTAS.md cuando terminemos.
- Leer artículo de los autores.
- Cambiar ruta de lectura de datos a "datos/"
- Cambiar nombre a fit.py y visualization.py para el que se decida.
- Añadir técnicas de selección/reducción dimensionalidad.
- Importancia de características cuando hagamos RF:
```python
importances = pipe_rf['clf'].feature_importances_
vs.plot_feature_importance(
    importances, 10,
    selection_strategy == Selection.PCA, SAVE_FIGURES, IMG_PATH)
```
- Utilizar importancia sin polinomios ni PCA en el conjunto de train para hacer gráficas de las dos más relevantes.
- Comentar código y revisar estilo
- Normalización vs estandarización?
- Comparar el mejor modelo de distintas clases de modelos. Gráficas.
- Escalar gráficas para que se vea algo. Eliminar outliers?
- Arreglar gráficas en general. Puntitos pequeños.
- Medir bien tiempos en varias etapas.
- Cambiar Readme.md si al final hacemos clasificación
