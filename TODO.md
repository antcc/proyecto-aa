# Al final
- Eliminar TODO.md, NOTAS.md y RESULTS.md cuando terminemos.
- Cambiar ruta de lectura de datos a "datos/"
- Quitar caché?
- Comentar código y revisar estilo

# Pendiente
- Añadir técnicas de reducción de dimensionalidad
- Importancia de características cuando hagamos RF. Utilizar importancia sin polinomios ni PCA en el conjunto de train para hacer gráficas de las dos más relevantes.
```python
importances = pipe_rf['clf'].feature_importances_
vs.plot_feature_importance(
    importances, 10,
    selection_strategy == Selection.PCA, SAVE_FIGURES, IMG_PATH)
```
- Comparar el mejor modelo de distintas clases de modelos. Gráficas.
- Arreglar gráficas en general (normalizar variables?).
