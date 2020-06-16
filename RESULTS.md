--- AJUSTE DE MODELOS LINEALES ---

-> PREANÁLISIS: constante de regularización

LR: 2.7 min
RC: 31s
SGD: 50s

-> AJUSTE

Tiempo para 35 candidatos: 1.912 min

--> Mejor clasificador encontrado <--
Modelo: SGDClassifier(alpha=0.004641588833612777, eta0=0.1, learning_rate='adaptive',
              random_state=2020)
Mejores Parámetros: {'clf__alpha': 0.004641588833612777, 'clf__learning_rate': 'adaptive'}
* Accuracy en CV: 64.866%

--- AJUSTE DE RANDOM FOREST ---

-> PREANÁLISIS: número de árboles y profundidad máxima

10 min

-> AJUSTE

Tiempo para 20 candidatos: 24.867 min

--> Mejor clasificador encontrado <--
Modelo: RandomForestClassifier(max_depth=20, n_estimators=600, random_state=2020)
Mejores Parámetros: {'clf__max_depth': 20, 'clf__n_estimators': 600}
* Accuracy en CV: 67.131%

--- AJUSTE DE MODELOS DE BOOSTING ---

-> PREANÁLISIS: número de árboles y profundidad máxima

ADABoost: 18 min
GradientBoostingClassifier: 14.4 min

-> AJUSTE

Tiempo para 63 candidatos: 71.794 min

Mostrando el top 3 de mejores modelos en validación cruzada...

GradientBoostingClassifier (ranking #1)
Parámetros: {'clf__learning_rate': 0.1, 'clf__max_depth': 2, 'clf__n_estimators': 325, 'clf__subsample': 0.75}
* Accuracy en CV: 67.227% (+- 0.579%)

GradientBoostingClassifier (ranking #2)
Parámetros: {'clf__learning_rate': 0.1, 'clf__max_depth': 2, 'clf__n_estimators': 300, 'clf__subsample': 0.75}
* Accuracy en CV: 67.196% (+- 0.476%)

GradientBoostingClassifier (ranking #3)
Parámetros: {'clf__learning_rate': 0.1, 'clf__max_depth': 3, 'clf__n_estimators': 275, 'clf__subsample': 1.0}
* Accuracy en CV: 67.151% (+- 0.242%)

--> Mejor clasificador encontrado <--
Modelo: GradientBoostingClassifier(max_depth=2, n_estimators=325, random_state=2020,
                           subsample=0.75)
Mejores Parámetros: {'clf__learning_rate': 0.1, 'clf__max_depth': 2, 'clf__n_estimators': 325, 'clf__subsample': 0.75}
* Accuracy en CV: 67.227%

--- AJUSTE DE MLP ---

-> PREANÁLISIS: número de neuronas por capa

19.6 min

-> AJUSTE

Tiempo para 20 candidatos: 23.942 min

Mostrando el top 3 de mejores modelos en validación cruzada...



--> Mejor clasificador encontrado <--
Modelo: MLPClassifier(alpha=1.6600481724236347, hidden_layer_sizes=(98, 98, 98),
              learning_rate='adaptive', learning_rate_init=0.1, max_iter=300,
              random_state=2020, solver='sgd', tol=0.001)
Mejores Parámetros: {'clf__alpha': 1.6600481724236347, 'clf__hidden_layer_sizes': (98, 98, 98)}
* Accuracy en CV: 66.147%

--- AJUSTE DE KNN ---

-> PREANÁLISIS: valor de k

46s

-> AJUSTE

Tiempo para 6 candidatos: 2.906 min

--> Mejor clasificador encontrado <--
Modelo: KNeighborsClassifier(n_neighbors=120, weights='distance')
Mejores Parámetros: {'clf__n_neighbors': 120, 'clf__weights': 'distance'}
* Accuracy en CV: 63.968%

--- AJUSTE DE REDES DE FUNCIONES DE BASE RADIAL ---

-> PREANÁLISIS: valor de k

27s

-> ANÁLISIS

Tiempo para 30 candidatos: 10.890 min

--> Mejor clasificador encontrado <--
Modelo: RBFNetworkClassifier(alpha=1.2915496650148826e-09, k=175, random_state=2020)
Mejores Parámetros: {'clf__alpha': 1.2915496650148826e-09, 'clf__k': 175}
* Accuracy en CV: 65.829%


# EN TEST

--- AJUSTE DE MODELO LINEAL ---

Entrenando clasificador lineal... Hecho.
Tiempo de entrenamiento: 0.276 min

--- AJUSTE DE RANDOM FOREST ---

Entrenando clasificador Random Forest... Hecho.
Tiempo de entrenamiento: 1.467 min

--- AJUSTE DE MODELO DE BOOSTING ---

Entrenando clasificador Gradient Boosting... Hecho.
Tiempo de entrenamiento: 1.575 min

--- AJUSTE DE MODELO MLP ---

Entrenando clasificador MLP... Hecho.
Tiempo de entrenamiento: 2.007 min

--- AJUSTE DE MODELO KNN ---

Entrenando clasificador KNN... Hecho.
Tiempo de entrenamiento: 0.013 min

--- AJUSTE DE MODELO DE REDES DE FUNCIONES DE BASE RADIAL ---

Entrenando clasificador RBF... Hecho.
Tiempo de entrenamiento: 0.770 min

--- AJUSTE DE MODELO ALEATORIO ---

Entrenando clasificador aleatorio... Hecho.
Tiempo de entrenamiento: 0.000 min

--- COMPARACIÓN DE LOS MEJORES MODELOS ---

--> SGDClassifier(alpha=0.004, eta0=0.1, learning_rate='adaptive',
              random_state=2020) <--
Número de variables usadas: 702
* Accuracy en training: 67.315%
* AUC en training: 73.676%
* Accuracy en test: 65.176%
* AUC en test: 70.460%

--> RandomForestClassifier(max_depth=20, n_estimators=600, n_jobs=-1,
                       random_state=2020) <--
Número de variables usadas: 58
* Accuracy en training: 99.798%
* AUC en training: 99.998%
* Accuracy en test: 66.714%
* AUC en test: 72.599%

--> GradientBoostingClassifier(max_depth=2, n_estimators=325, random_state=2020,
                           subsample=0.75) <--
Número de variables usadas: 58
* Accuracy en training: 69.470%
* AUC en training: 76.431%
* Accuracy en test: 66.386%
* AUC en test: 72.654%

--> MLPClassifier(alpha=1.5, hidden_layer_sizes=(99, 99), learning_rate='adaptive',
              learning_rate_init=0.1, random_state=2020, solver='sgd',
              tol=0.001) <--
Número de variables usadas: 58
* Accuracy en training: 67.481%
* AUC en training: 73.709%
* Accuracy en test: 65.428%
* AUC en test: 71.406%

--> KNeighborsClassifier(n_jobs=-1, n_neighbors=120, weights='distance') <--
Número de variables usadas: 58
* Accuracy en training: 100.000%
* AUC en training: 100.000%
* Accuracy en test: 63.915%
* AUC en test: 68.784%

--> RBFNetworkClassifier(alpha=1e-09, k=175, random_state=2020) <--
Número de variables usadas: 58
* Accuracy en training: 65.874%
* AUC en training: 71.224%
* Accuracy en test: 64.772%
* AUC en test: 69.998%

--> DummyClassifier(strategy='stratified') <--
Número de variables usadas: 58
* Accuracy en training: 50.425%
* AUC en training: 50.084%
* Accuracy en test: 50.580%
* AUC en test: 49.777%

Tiempo total de ejecución: 13.848 min
