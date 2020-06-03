# Modelos lineales

--> PCA(0.95)

Tiempo para 18 candidatos: 10.180m

Mostrando el top 3 de mejores modelos en validación cruzada...

RidgeClassifier (ranking #1)
Parámetros: {'clf__alpha': 1.0}
* Accuracy en CV: 65.575% (+- 0.473%)

RidgeClassifier (ranking #2)
Parámetros: {'clf__alpha': 10.0}
* Accuracy en CV: 65.560% (+- 0.507%)

RidgeClassifier (ranking #3)
Parámetros: {'clf__alpha': 0.001}
* Accuracy en CV: 65.539% (+- 0.620%)

--> Mejor clasificador encontrado <--
Modelo: RidgeClassifier(max_iter=5000, random_state=2020)
Mejores Parámetros: {'clf__alpha': 1.0}
Número de variables usadas: 702
* Accuracy en CV: 65.575%
* Accuracy en training: 67.542%
* AUC en training: 74.155%
* Accuracy en test: 65.596%
* AUC en test: 70.850%

# Random Forest

Tiempo para 20 candidatos: 17.538m

Mostrando el top 3 de mejores modelos en validación cruzada...

RandomForestClassifier (ranking #1)
Parámetros: {'clf__ccp_alpha': 0.00017970243840241938, 'clf__max_depth': 29, 'clf__n_estimators': 200}
* Accuracy en CV: 67.117% (+- 0.366%)

RandomForestClassifier (ranking #2)
Parámetros: {'clf__ccp_alpha': 0.0002298529412734188, 'clf__max_depth': None, 'clf__n_estimators': 400}
* Accuracy en CV: 67.077% (+- 0.353%)

RandomForestClassifier (ranking #3)
Parámetros: {'clf__ccp_alpha': 5.2985619275654474e-05, 'clf__max_depth': 29, 'clf__n_estimators': 200}
* Accuracy en CV: 66.980% (+- 0.408%)

--> Mejor clasificador encontrado <--
Modelo: RandomForestClassifier(ccp_alpha=0.00017970243840241938, max_depth=29,
                               n_estimators=200, n_jobs=-1, random_state=2020)
Mejores Parámetros: {'clf__ccp_alpha': 0.00017970243840241938, 'clf__max_depth': 29, 'clf__n_estimators': 200}
Número de variables usadas: 58
* Accuracy en CV: 67.117%
* Accuracy en training: 82.036%
* AUC en training: 90.297%
* Accuracy en test: 66.420%
* AUC en test: 72.560%

# Dummy

Número de variables usadas: 58
* Accuracy en training: 50.105%
* AUC en training: 49.398%
* Accuracy en test: 49.983%
* AUC en test: 50.764%
