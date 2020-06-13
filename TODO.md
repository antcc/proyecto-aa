# Al final
- Eliminar TODO.md, NOTAS.md y RESULTS.md cuando terminemos.
- Quitar verbose = 1 de GSearch y RSearch
- Cambiar ruta de lectura de datos a "datos/"
- Quitar caché?
- Comentar código y revisar estilo

# Pendiente
- Añadir técnicas de reducción de dimensionalidad
- Normalizar las gráficas para que se vean cosas. Quitar outliers.
- Importancia de características cuando hagamos RF. Utilizar importancia sin polinomios ni PCA en el conjunto de train para hacer gráficas de las dos más relevantes.
```python
importances = pipe_rf['clf'].feature_importances_
vs.plot_feature_importance(
    importances, 10,
    selection_strategy == Selection.PCA, SAVE_FIGURES, IMG_PATH)
```
- Comparar el mejor modelo de distintas clases de modelos. Gráficas.
- Arreglar gráficas en general (normalizar variables?).

PONER CÓDIGO PARA QUE SE ENTRENEN Y EVALUEN SOLO LOS MEJORES MODELOS, Y LO DE ELEGIR LOS PARÁMETROS SEA OPCIONAL.
- Fallo de memoria con rbf.
