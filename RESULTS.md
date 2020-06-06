# Modelos lineales

--> PCA(0.95) + Poly(2)

Tiempo para 18 candidatos: 6.520min

Mostrando el top 3 de mejores modelos en validación cruzada...

RidgeClassifier (ranking #1)
Parámetros: {'clf__alpha': 1.7782794100389228}
* Accuracy en CV: 65.571% (+- 0.441%)

RidgeClassifier (ranking #2)
Parámetros: {'clf__alpha': 10.0}
* Accuracy en CV: 65.560% (+- 0.507%)

RidgeClassifier (ranking #3)
Parámetros: {'clf__alpha': 0.05623413251903491}
* Accuracy en CV: 65.535% (+- 0.643%)

--> Mejor clasificador encontrado <--
Modelo: RidgeClassifier(alpha=1.7782794100389228, max_iter=3000, random_state=2020)
Mejores Parámetros: {'clf__alpha': 1.7782794100389228}
Número de variables usadas: 702
* Accuracy en CV: 65.571%
* Accuracy en training: 67.560%
* AUC en training: 74.137%
* Accuracy en test: **65.630%**
* AUC en test: **70.853%**


# Random Forest

Tiempo para 20 candidatos: 18.802m

Mostrando el top 3 de mejores modelos en validación cruzada...

RandomForestClassifier (ranking #1)
Parámetros: {'clf__ccp_alpha': 0.00013188070892093489, 'clf__max_depth': None, 'clf__n_estimators': 400}
* Accuracy en CV: 67.373% (+- 0.288%)

RandomForestClassifier (ranking #2)
Parámetros: {'clf__ccp_alpha': 9.10352215253087e-05, 'clf__max_depth': 29, 'clf__n_estimators': 400}
* Accuracy en CV: 67.240% (+- 0.484%)

RandomForestClassifier (ranking #3)
Parámetros: {'clf__ccp_alpha': 3.229291988920536e-05, 'clf__max_depth': 29, 'clf__n_estimators': 400}
* Accuracy en CV: 67.182% (+- 0.477%)

--> Mejor clasificador encontrado <--
Modelo: RandomForestClassifier(n_jobs=-1, random_state=2020)
Mejores Parámetros: {'clf__ccp_alpha': 0.00013188070892093489, 'clf__max_depth': None, 'clf__n_estimators': 400}
Número de variables usadas: 58
* Accuracy en CV: 67.373%
* Accuracy en training: 90.674%
* AUC en training: 97.091%
* Accuracy en test: **66.479%**
* AUC en test: **72.543%**

# Dummy

Número de variables usadas: 58
* Accuracy en training: 50.105%
* AUC en training: 49.398%
* Accuracy en test: **49.983%**
* AUC en test: **50.764%**
