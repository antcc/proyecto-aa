------- PROYECTO FINAL: AJUSTE DE MODELOS DE CLASIFICACIÓN -------

Leyendo datos de OnlineNewsPopularity.csv... Hecho.

--- AJUSTE DE MODELO LINEAL ---

Entrenando clasificador lineal... Hecho.
Tiempo de entrenamiento: 0.139 min

--- AJUSTE DE RANDOM FOREST ---

Entrenando clasificador Random Forest... Hecho.
Tiempo de entrenamiento: 0.511 min

--- AJUSTE DE MODELO DE BOOSTING ---

Entrenando clasificador Gradient Boosting... Hecho.
Tiempo de entrenamiento: 0.503 min

--- AJUSTE DE MODELO MLP ---

Entrenando clasificador MLP... Hecho.
Tiempo de entrenamiento: 0.769 min

--- AJUSTE DE MODELO KNN ---

Entrenando clasificador KNN... Hecho.
Tiempo de entrenamiento: 0.006 min

--- AJUSTE DE MODELO DE REDES DE FUNCIONES DE BASE RADIAL ---

Entrenando clasificador RBF... Hecho.
Tiempo de entrenamiento: 0.377 min

--- AJUSTE DE MODELO ALEATORIO ---

Entrenando clasificador aleatorio... Hecho.
Tiempo de entrenamiento: 0.000 min

--- COMPARACIÓN DE LOS MEJORES MODELOS ---

--> LogisticRegression(C=0.1, max_iter=1000, random_state=2020) <--
Número de variables usadas: 702
* Accuracy en training: 67.737%
* AUC en training: 74.321%
* Accuracy en test: 65.546%
* AUC en test: 70.919%

--> RandomForestClassifier(max_depth=20, n_estimators=600, n_jobs=-1,
                       random_state=2020) <--
Número de variables usadas: 58
* Accuracy en training: 99.798%
* AUC en training: 99.998%
* Accuracy en test: 66.714%
* AUC en test: 72.599%

--> GradientBoostingClassifier(max_depth=4, random_state=2020, subsample=0.75) <--
Número de variables usadas: 58
* Accuracy en training: 71.207%
* AUC en training: 78.613%
* Accuracy en test: 66.437%
* AUC en test: 72.835%

--> MLPClassifier(alpha=3.0, hidden_layer_sizes=(88, 88), learning_rate='adaptive',
              learning_rate_init=0.1, random_state=2020, solver='sgd',
              tol=0.001) <--
Número de variables usadas: 58
* Accuracy en training: 66.040%
* AUC en training: 71.770%
* Accuracy en test: 64.831%
* AUC en test: 70.251%

--> KNeighborsClassifier(n_jobs=-1, n_neighbors=150, weights='distance') <--
Número de variables usadas: 58
* Accuracy en test: 63.948%
* AUC en test: 68.798%

--> RBFNetworkClassifier(alpha=1e-10, k=300, random_state=2020) <--
Número de variables usadas: 58
* Accuracy en training: 65.928%
* AUC en training: 71.396%
* Accuracy en test: 64.831%
* AUC en test: 70.135%

--> DummyClassifier(strategy='stratified') <--
Número de variables usadas: 58
* Accuracy en training: 50.425%
* AUC en training: 50.084%
* Accuracy en test: 50.580%
* AUC en test: 49.777%

Tiempo total de ejecución: 3.457 min
