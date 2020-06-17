# Dataset

URL: https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity#
This dataset summarizes a heterogeneous set of features about articles published by Mashable in a period of two years.
39644 instances with 60 attributes (Integer, Real). 2 de ellos no predictivos (los dos primeros, URL y el timestamp). Es decir, 58 atributos.
Target: no. of shares in social networks (Integer)
NO HAY MISSING VALUES (YAY!)

Si se enfoca como clasificación binaria: distribución de clases
Shares Value Range:   Number of Instances in Range:
   <  1400            18490
   >= 1400            21154

Binary classification as popular vs unpopular using a decision, threshold of 1400 social interactions. Recorded 67% of accuracy and 0.73 of AUC. Usando RandomForestClassifier.
Experiments with different models: Random Forest (best model),
             Adaboost, SVM, KNN and Naïve Bayes.

Summary Statistics:
                       Feature       Min          Max         Mean           SD
                     timedelta    8.0000     731.0000     354.5305     214.1611
                n_tokens_title    2.0000      23.0000      10.3987       2.1140
              n_tokens_content    0.0000    8474.0000     546.5147     471.1016
               n_unique_tokens    0.0000     701.0000       0.5482       3.5207
              n_non_stop_words    0.0000    1042.0000       0.9965       5.2312
      n_non_stop_unique_tokens    0.0000     650.0000       0.6892       3.2648
                     num_hrefs    0.0000     304.0000      10.8837      11.3319
                num_self_hrefs    0.0000     116.0000       3.2936       3.8551
                      num_imgs    0.0000     128.0000       4.5441       8.3093
                    num_videos    0.0000      91.0000       1.2499       4.1078
          average_token_length    0.0000       8.0415       4.5482       0.8444
                  num_keywords    1.0000      10.0000       7.2238       1.9091
     data_channel_is_lifestyle    0.0000       1.0000       0.0529       0.2239
 data_channel_is_entertainment    0.0000       1.0000       0.1780       0.3825
           data_channel_is_bus    0.0000       1.0000       0.1579       0.3646
        data_channel_is_socmed    0.0000       1.0000       0.0586       0.2349
          data_channel_is_tech    0.0000       1.0000       0.1853       0.3885
         data_channel_is_world    0.0000       1.0000       0.2126       0.4091
                    kw_min_min   -1.0000     377.0000      26.1068      69.6323
                    kw_max_min    0.0000  298400.0000    1153.9517    3857.9422
                    kw_avg_min   -1.0000   42827.8571     312.3670     620.7761
                    kw_min_max    0.0000  843300.0000   13612.3541   57985.2980
                    kw_max_max    0.0000  843300.0000  752324.0667  214499.4242
                    kw_avg_max    0.0000  843300.0000  259281.9381  135100.5433
                    kw_min_avg   -1.0000    3613.0398    1117.1466    1137.4426
                    kw_max_avg    0.0000  298400.0000    5657.2112    6098.7950
                    kw_avg_avg    0.0000   43567.6599    3135.8586    1318.1338
     self_reference_min_shares    0.0000  843300.0000    3998.7554   19738.4216
     self_reference_max_shares    0.0000  843300.0000   10329.2127   41027.0592
    self_reference_avg_sharess    0.0000  843300.0000    6401.6976   24211.0269
             weekday_is_monday    0.0000       1.0000       0.1680       0.3739
            weekday_is_tuesday    0.0000       1.0000       0.1864       0.3894
          weekday_is_wednesday    0.0000       1.0000       0.1875       0.3903
           weekday_is_thursday    0.0000       1.0000       0.1833       0.3869
             weekday_is_friday    0.0000       1.0000       0.1438       0.3509
           weekday_is_saturday    0.0000       1.0000       0.0619       0.2409
             weekday_is_sunday    0.0000       1.0000       0.0690       0.2535
                    is_weekend    0.0000       1.0000       0.1309       0.3373
                        LDA_00    0.0000       0.9270       0.1846       0.2630
                        LDA_01    0.0000       0.9259       0.1413       0.2197
                        LDA_02    0.0000       0.9200       0.2163       0.2821
                        LDA_03    0.0000       0.9265       0.2238       0.2952
                        LDA_04    0.0000       0.9272       0.2340       0.2892
           global_subjectivity    0.0000       1.0000       0.4434       0.1167
     global_sentiment_polarity   -0.3937       0.7278       0.1193       0.0969
    global_rate_positive_words    0.0000       0.1555       0.0396       0.0174
    global_rate_negative_words    0.0000       0.1849       0.0166       0.0108
           rate_positive_words    0.0000       1.0000       0.6822       0.1902
           rate_negative_words    0.0000       1.0000       0.2879       0.1562
         avg_positive_polarity    0.0000       1.0000       0.3538       0.1045
         min_positive_polarity    0.0000       1.0000       0.0954       0.0713
         max_positive_polarity    0.0000       1.0000       0.7567       0.2478
         avg_negative_polarity   -1.0000       0.0000      -0.2595       0.1277
         min_negative_polarity   -1.0000       0.0000      -0.5219       0.2903
         max_negative_polarity   -1.0000       0.0000      -0.1075       0.0954
            title_subjectivity    0.0000       1.0000       0.2824       0.3242
      title_sentiment_polarity   -1.0000       1.0000       0.0714       0.2654
        abs_title_subjectivity    0.0000       0.5000       0.3418       0.1888
  abs_title_sentiment_polarity    0.0000       1.0000       0.1561       0.2263

# Modelos

Es necesario comparar UN modelo lineal y al menos DOS de los siguientes; en cada caso hay que encontrar el mejor posible dentro de su clase. Cada modelo extra sobre esa base proporciona 5 puntos como máximo.

- **Perceptron Multicapa**: Considerar clases de funciones definida por arquitecturas con 3 capas y un número de unidades por capa en el rango 50-100. Considerar el número de neuronas por capa como un hiperparámetro.

- **Boosting**: Se recomienda que para clasificación se usen funciones “stamp”.

- **Random Forest**: Usar como hiperparámetros los valores que por defecto se dan en teoría y experimentar para obtener el número de árboles adecuado.

- **Red de Funciones de base Radial**: Hay que fijar el valor K en el modelo final. Pero se deben evaluar distintos valores de K como criterio para la elección del K final.

# Pasos a seguir

2. Argumentos a favor de la elección de los modelos.
6. Justificación de la función de pérdida usada.
7. Selección de las técnica (parámetrica) y valoración de la idoneidad de la misma frente a otras alternativas
8. Aplicación de la técnica especificando claramente que algoritmos se usan en la estimación de los parámetros, los hiperparámetros y el error de generalización.
9. Argumentar sobre la idoneidad de la función regularización usada (en su caso)
10. Valoración de los resultados ( gráficas, métricas de error, análisis de residuos, etc )
11. Justificar que se ha obtenido la mejor de las posibles soluciones con la técnica elegida y la muestra dada. Argumentar en términos de los errores de ajuste y generalización,

# Consideraciones

- Estudiar distribución de clases. Concluir que no están desbalanceadas. (ver [Dataset])
- Justificar elección de funciones y de TODOS los parámetros (incluidos por defecto).
- Explicar con detalle el funcionamiento de pipelines y gridSearch.
- Pueden usarse técnicas de reducción de dimensionalidad, ej. PCA o Random Projection si se justifica que su uso mejora los resultados.

- Compilación con `make` en la carpeta `doc`. Compilación continua:
```
while true; do make; inotifywait -e modify,close_write memoria.md; done
```
- **Importante:** poner versión de sklearn en la memoria.




# Enfoque

- MOSTRAR ANÁLISIS ESTADÍSTICO DEL PAPER
- Leer artículo de los autores. Repetir experimentos que hacen.
- Mirar los parámetros por defecto que ponen.
- Mirar las métricas que usan. usar varias métricas.
- Gráfica de acc en CV en función de alguna cosa (regularización por ej)
- Comentar que los valores de ccp_alpha para RandomForest se han obtenido llamando a
`DecisionTreeClassifier(max_depth = 20).cost_complexity_pruning_path(X_train, y_train)["ccp_alphas"]`. El criterio de selección de predictores por defecto es sqrt.
- Ventajas e inconvenientes de cada modelo. Pérdida de interpretabilidad, tiempo, etc.
- Modelos lineales: llegan a casi lo mismo que otros muchos más potentes.
- Elegir el modelo en cada caso: si hay variables numéricas, elijo tal, si no, cual..

# Formato de la documentación

- Rellenar el `README.md`
- Rellenar anexo con funcionamiento del código? Listings?
- Paquetes de latex en `header.md`
- Números con dólares.
