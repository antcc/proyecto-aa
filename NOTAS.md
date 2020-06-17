

# Pasos a seguir

2. Argumentos a favor de la elección de los modelos.
6. Justificación de la función de pérdida usada.
9. Argumentar sobre la idoneidad de la función regularización usada (en su caso)
10. Valoración de los resultados ( gráficas, métricas de error, análisis de residuos, etc )
11. Justificar que se ha obtenido la mejor de las posibles soluciones con la técnica elegida y la muestra dada. Argumentar en términos de los errores de ajuste y generalización,

# Enfoque

- Comentar que los valores de ccp_alpha para RandomForest se han obtenido llamando a
`DecisionTreeClassifier(max_depth = 20).cost_complexity_pruning_path(X_train, y_train)["ccp_alphas"]`. El criterio de selección de predictores por defecto es sqrt.
- Ventajas e inconvenientes de cada modelo. Pérdida de interpretabilidad, tiempo, etc. Argumentos a favor de los modelos. (variables nuimércias, balñanceo...)
- Modelos lineales: llegan a casi lo mismo que otros muchos más potentes.
- Elegir el modelo en cada caso: si hay variables numéricas, elijo tal, si no, cual..
- Comparar tabla de resultados con resuklts.md
# Formato de la documentación

- Rellenar el `README.md`
- Rellenar anexo con funcionamiento del código
