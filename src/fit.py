#!/usr/bin/env python
# coding: utf-8
# uso: ./fit.py

"""
Aprendizaje Automático. Curso 2019/20.
Proyecto final: ajuste del mejor modelo de clasificación.
Intentamos conseguir el mejor ajuste posible dentro de una clase
acotada de modelos para un problema de clasificación binaria.

Base de datos: Online News Popularity
https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity

Miguel Lentisco Ballesteros
Antonio Coín Castro
"""

#
# LIBRERÍAS
#

import numpy as np
from timeit import default_timer
from pandas import read_csv
from enum import Enum
from joblib import Memory
from shutil import rmtree

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.random_projection import SparseRandomProjection

import visualization as vs

#
# CLASES Y ESTRUCTURAS GLOBALES
#

class Show(Enum):
    """Cantidad de gráficas a mostrar."""

    NONE = 0
    SOME = 1
    ALL = 2

class Selection(Enum):
    """Estrategia de selección de características."""

    PCA = 0
    RANDOM_PROJECTION = 1
    NONE = 2

class Model(Enum):
    """Clases de modelos para los ajustes."""

    LINEAR = 0
    TREES = 1
    MLP = 2

#
# PARÁMETROS GLOBALES
#

SEED = 2020
N_CLASSES = 2
CLASS_THRESHOLD = 1400
PATH = "../datos/"
DATASET_NAME = "OnlineNewsPopularity.csv"
CACHEDIR = "cachedir"
SHOW_CV_RESULTS = True
SAVE_FIGURES = False
IMG_PATH = "../doc/img"
SHOW = Show.SOME

#
# FUNCIONES AUXILIARES
#

def print_evaluation_metrics(clf, X_lst, y_lst, names):
    """Imprime la evaluación de resultados en varios conjuntos de un clasificador.
      - X_lst: lista de matrices de características.
      - y_lst: lista de vectores de etiquetas.
      - names: lista de nombres de los conjuntos (training, test, ...)."""

    for name, X, y in zip(names, X_lst, y_lst):
        print("* Accuracy en {}: {:.3f}%".format(
            name, 100.0 * clf.score(X, y)))

def print_cv_metrics(results, n_top = 3):
    """Imprime un resumen de los resultados obtenidos en validación cruzada.
         - results: diccionario de resultados.
         - n_top: número de modelos a mostrar de entre los mejores."""

    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            model = results['params'][candidate]
            print("{} (ranking #{})".format(
                results['params'][candidate]['clf'].__class__.__name__, i))
            print("Parámetros: {}".format(
                {k: model[k] for k in model.keys() if k != 'clf'}))
            print("* Accuracy en CV: {:.3f}% (+- {:.3f}%)\n".format(
                100.0 * results['mean_test_score'][candidate],
                100.0 * results['std_test_score'][candidate]))

#
# LECTURA Y MANIPULACIÓN DE DATOS
#

def read_data(filename):
    """Carga los datos de fichero, elimina los atributos no predictivos, y separa
       los predictores, las etiquetas y el nombre de los atributos."""

    # Cargamos los datos quitando las dos primeras columnas
    df = read_csv(
        filename,
        sep = ', ',
        engine = 'python',
        header = 0,
        usecols = [i for i in range(2, 62)],
        index_col = False,
        dtype = np.float64)

    # Convertimos la última columna en etiquetas binarias
    df.iloc[:, -1] = df.iloc[:, -1].apply(
        lambda x: -1.0 if x < CLASS_THRESHOLD else 1.0)

    # Separamos predictores, objetivos y nombres
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    names = df.columns.values

    return X, y, names

def split_data(X, y, val_size = 0.0, test_size = 0.3):
    """Realiza una división de los datos en entrenamiento/validación/test
       según las proporciones indicadas."""

    # Divisón para entrenamiento
    X_train, X_test, y_train, y_test = \
        train_test_split(
            X, y,
            test_size = test_size,
            stratify = y,
            random_state = SEED)

    # Divisón para validación
    if val_size > 0.0:
        X_train, X_val, y_train, y_val = \
            train_test_split(
                X_train, y_train,
                test_size = val_size / (1.0 - test_size),
                stratify = y_train,
                random_state = SEED)
    else:
        X_val = []
        y_val = []

    return X_train, X_val, X_test, y_train, y_val, y_test

#
# PREPROCESADO DE DATOS
#

def preprocess_pipeline(model_class, selection_strategy = Selection.PCA):
    """Construye una lista de transformaciones para el
       preprocesamiento de datos, incluyendo transformaciones polinómicas
       y selección de características. Hay diferentes estrategias para cada
       clase de modelos."""

    preproc_params = {}

    if model_class == Model.LINEAR:
        if selection_strategy == Selection.PCA:
            preproc = [
                ("var", VarianceThreshold()),
                ("standardize", StandardScaler()),
                ("selection", PCA(0.95, random_state = SEED)),
                ("poly", PolynomialFeatures(2, include_bias = False)),
                ("standardize2", StandardScaler())]

        elif selection_strategy == Selection.RANDOM_PROJECTION:
            preproc = [
                ("var", VarianceThreshold()),
                ("standardize", StandardScaler()),
                ("selection", SparseRandomProjection(39, random_state = SEED)),
                ("poly", PolynomialFeatures(2, include_bias = False)),
                ("standardize2", StandardScaler())]

        else:
            preproc = [
                ("var", VarianceThreshold()),
                ("standardize", StandardScaler()),
                ("poly", PolynomialFeatures(2, include_bias = False)),
                ("standardize2", StandardScaler())]

    return preproc, preproc_params

def fit(X_train, X_test, y_train, y_test, clfs, model_class, selection_strategy):
    """Ajuste de varios modelos de clasificación para un conjunto de datos, eligiendo
       por validación cruzada el mejor de ellos dentro de una clase concreta.
         - X_train, y_train: datos de entrenamiento.
         - X_test, y_test: datos de test.
         - selection_strategy: estrategia de selección de características (ver 'Selection').
         - model: clase de modelos a ajustar (ver 'Model').
         - clfs: lista de diccionarios con los modelos concretos para ajustar.
         """

    # Construimos un pipeline de preprocesado + clasificación (placeholder) con caché
    preproc, preproc_params = preprocess_pipeline(model_class, selection_strategy)
    memory = Memory(location = CACHEDIR, verbose = 0)
    pipe = Pipeline(preproc + [("clf", DummyClassifier())], memory = memory)
    search_space = [{**preproc_params, **params} for params in clfs]

    # Buscamos los mejores modelos por CV
    print("Comparando modelos por validación cruzada... ", end = "", flush = True)
    start = default_timer()
    best_clf = GridSearchCV(
        pipe, search_space,
        scoring = 'accuracy',
        cv = 5, n_jobs = -1)
    best_clf.fit(X_train, y_train)
    elapsed = default_timer() - start
    print("Hecho.")
    print("Tiempo para {} candidatos: {:.3f}s\n".format(
        len(best_clf.cv_results_['params']), elapsed))

    if SHOW_CV_RESULTS:
        # Mostramos los resultados de CV
        n_top = 3
        print("Mostrando el top {} de mejores modelos en validación cruzada...\n"
            .format(n_top))
        print_cv_metrics(best_clf.cv_results_, n_top)

    # Mostramos los resultados del mejor clasificador encontrado
    model = best_clf.best_params_
    print("--> Mejor clasificador encontrado <--")
    print("Modelo: {}".format(model['clf']))
    print("Mejores Parámetros: {}".format(
        {k: model[k] for k in model.keys() if k != 'clf'}))
    print("Número de variables usadas: {}".format(
        best_clf.best_estimator_['clf'].coef_.shape[1]))
    print("* Accuracy en CV: {:.3f}%".format(100.0 * best_clf.best_score_))
    print_evaluation_metrics(
        best_clf,
        [X_train, X_test],
        [y_train, y_test],
        ["training", "test"])
    print("")

    if SHOW != Show.NONE:
        vs.wait()

        # Obtenemos los datos preprocesados para las gráficas
        preproc_pipe = Pipeline(best_clf.best_estimator_.steps[:-1])
        X_train_pre = preproc_pipe.transform(X_train)
        X_test_pre = preproc_pipe.transform(X_test)

        print("Mostrando matriz de correlación antes y después del preprocesado...")
        vs.plot_corr_matrix(X_train, X_train_pre, SAVE_FIGURES, IMG_PATH)

        # Matriz de confusión
        print("Mostrando matriz de confusión en test...")
        vs.confusion_matrix(best_clf, X_test, y_test, SAVE_FIGURES, IMG_PATH)

        if selection_strategy == Selection.PCA:
            # Predicciones para el conjunto de test
            y_pred = best_clf.predict(X_test)

            print("Mostrando proyección de las dos primeras componentes principales en test "
                  "con etiquetas predichas...")
            vs.scatter_pca(X_test_pre, y_pred, SAVE_FIGURES, IMG_PATH)

        if SHOW == Show.ALL:
            print("Calculando y mostrando curva de aprendizaje... ")
            vs.plot_learning_curve(
                best_clf,
                X_train, y_train,
                n_jobs = -1, cv = 5,
                scoring = 'accuracy',
                save_figures = SAVE_FIGURES,
                img_path = IMG_PATH)

    # Limpiamos la caché
    memory.clear(warn = False)

    return best_clf

def fit_dummy(X_train, X_test, y_train, y_test):
    """Ajustamos un clasificador que estima una clase aleatoria teniendo en
       cuenta la distribución de clases."""

    # Elegimos un clasificador aleatorio
    dummy_clf = DummyClassifier(strategy = 'stratified')

    # Ajustamos el modelo
    dummy_clf.fit(X_train, y_train)

    # Mostramos los resultados
    print("Número de variables usadas: {}".format(X_train.shape[1]))
    print_evaluation_metrics(
        dummy_clf,
        [X_train, X_test],
        [y_train, y_test],
        ["training", "test"])
    print("")

    return dummy_clf

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Función principal. Ejecuta el proyecto paso a paso."""

    start = default_timer()

    # Semilla aleatoria para reproducibilidad
    np.random.seed(SEED)

    # Número de decimales fijo para salida de vectores
    np.set_printoptions(formatter = {'float': lambda x: "{:0.3f}".format(x)})

    print("------- PROYECTO FINAL: AJUSTE DE MODELOS DE CLASIFICACIÓN -------\n")

    #
    # LECTURA DE DATOS
    #

    # Cargamos los datos de entrenamiento y test
    print("Cargando datos de entrenamiento y test... ", end = "", flush = True)
    X, y, attr_names = read_data(PATH + DATASET_NAME)
    X_train, X_val, X_test, y_train, y_val, y_test = \
        split_data(X, y, val_size = 0.0, test_size = 0.3)
    print("Hecho.\n")

    #
    # INSPECCIÓN DE LOS DATOS
    #

    if SHOW != Show.NONE:
        print("--- VISUALIZACIÓN DE LOS DATOS ---\n")

        # Mostramos distribución de clases en training y test
        print("Mostrando gráfica de distribución de clases...")
        vs.plot_class_distribution(y_train, y_test, N_CLASSES, SAVE_FIGURES, IMG_PATH)

        # Visualizamos las variables más relevantes
        print("Mostrando proyección de las dos variables más relevantes...")
        # TODO: elegir las más relevantes por criterio RF
        features = [24, 25]
        vs.plot_features(
            features, attr_names[features],
            X_train, y_train,
            SAVE_FIGURES, IMG_PATH)

        if SHOW == Show.ALL:
            # Visualizamos el conjunto de entrenamiento en 2 dimensiones
            print("Mostrando proyección del conjunto de entrenamiento en dos dimensiones...")
            vs.plot_tsne(X_train, y_train, SAVE_FIGURES, IMG_PATH)

    #
    # AJUSTE DE MODELOS LINEALES
    #

    print("--- AJUSTE DE MODELOS LINEALES ---\n")

    # Escogemos modelos lineales
    max_iter = 3000
    clfs_lin = [
        {"clf": [LogisticRegression(penalty = 'l2',
                                    max_iter = max_iter,
                                    class_weight = 'balanced')],
         "clf__C": np.logspace(-4, 4, 5)},
        {"clf": [RidgeClassifier(random_state = SEED,
                                 class_weight = 'balanced',
                                 max_iter = max_iter)],
         "clf__alpha": np.logspace(-4, 4, 5)}]

    # Ajustamos el mejor modelo
    best_clf_lin = fit(
        X_train, X_test,
        y_train, y_test,
        clfs = clfs_lin,
        selection_strategy = Selection.PCA,
        model_class = Model.LINEAR)

    #
    # CLASIFICADOR ALEATORIO
    #

    print("--- AJUSTE DE MODELO ALEATORIO ---\n")

    clf_dummy = fit_dummy(X_train, X_test, y_train, y_test)

    # Imprimimos tiempo total de ejecución
    elapsed = default_timer() - start
    print("Tiempo total de ejecución: {:.3f}s".format(elapsed))

    # Eliminamos directorio de caché
    rmtree(CACHEDIR)

if __name__ == "__main__":
    main()
