#!/usr/bin/env python
# coding: utf-8
# uso: ./fit_class.py

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

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import visualization_class as vs

#
# PARÁMETROS Y DEFINICIONES GLOBALES
#

SEED = 2020
N_CLASSES = 2
CLASS_THRESHOLD = 1400
PATH = "../datos/"
DATASET_NAME = "OnlineNewsPopularity.csv"
SAVE_FIGURES = False
IMG_PATH = "../doc/img"
SHOW = 1  # 0: Ninguna gráfica; 1: algunas gráficas; 2: todas las gráficas

class Selection(Enum):
    """Estrategia de selección de características."""

    PCA = 0
    NONE = 1

class Model(Enum):
    """Clases de modelos para los ajustes."""

    LINEAR = 0
    TREES = 1
    MLP = 2

#
# FUNCIONES AUXILIARES
#

def print_evaluation_metrics(clf, X_lst, y_lst, names):
    """Imprime la evaluación de resultados en varios conjuntos de un clasificador.
      - X_lst: lista de matrices de características.
      - y_lst: lista de vectores de etiquetas.
      - names: lista de nombres de los conjuntos (training, test, ...)."""

    for name, X, y in zip(names, X_lst, y_lst):
        print("Accuracy en {}: {:.3f}%".format(
            name, 100.0 * clf.score(X, y)))

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
    df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: -1.0 if x < CLASS_THRESHOLD else 1.0)

    # Separamos predictores, objetivos y nombres
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    names = df.columns.values

    return X, y, names

def split_data(X, y, val_size = 0.2, test_size = 0.3):
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

def preprocess_pipeline(model, selection_strategy = Selection.PCA):
    """Construye una lista de transformaciones para el
       preprocesamiento de datos, incluyendo transformaciones polinómicas
       y selección de características. Hay diferentes estrategias para cada
       clase de modelos."""

    if model == Model.LINEAR:
        if selection_strategy == Selection.PCA:
            preproc = [
                ("var", VarianceThreshold()),
                ("standardize", StandardScaler()),
                ("selection", PCA(0.90)),
                ("poly", PolynomialFeatures(2, include_bias = False)),
                ("standardize2", StandardScaler())]
        else:
            preproc = [
                ("var", VarianceThreshold()),
                ("standardize", StandardScaler()),
                ("poly", PolynomialFeatures(2, include_bias = False)),
                ("standardize2", StandardScaler())]

    return preproc

def fit_linear(X_train, X_test, y_train, y_test, selection_strategy = Selection.PCA):
    """Ajuste de un modelo lineal para un conjunto de datos."""

    # Creamos un pipeline de preprocesado
    preproc = preprocess_pipeline(Model.LINEAR, selection_strategy)
    preproc_pipe = Pipeline(preproc)

    # Obtenemos los datos preprocesados por si los necesitamos
    X_train_pre = preproc_pipe.fit_transform(X_train, y_train)
    X_test_pre = preproc_pipe.transform(X_test)

    # Construimos un pipeline de preprocesado + clasificación
    # (el clasificador que ponemos puede ser cualquiera, es un 'placeholder')
    pipe = Pipeline(preproc + [("clf", LogisticRegression())])

    if SHOW > 0:
        print("\nMostrando gráficas sobre preprocesado y características...")

        # Mostramos matriz de correlación de training antes y después de preprocesado
        vs.plot_corr_matrix(X_train, X_train_pre, SAVE_FIGURES, IMG_PATH)

    # Elegimos los modelos lineales y sus parámetros para CV
    max_iter = 1000
    search_space = [
        {"clf": [LogisticRegression(penalty = 'l2',
                                    max_iter = max_iter)],
         "clf__C": np.logspace(-4, 4, 5)},
        {"clf": [RidgeClassifier(random_state = SEED,
                                 max_iter = max_iter)],
         "clf__alpha": np.logspace(-4, 4, 5)},
        {"clf": [Perceptron(penalty = 'l2',
                            random_state = SEED,
                            max_iter = max_iter)],
         "clf__alpha": np.logspace(-4, 4, 5)}]

    # Buscamos los mejores parámetros por CV
    print("Ajustando un modelo lineal...\n")
    start = default_timer()
    best_clf = GridSearchCV(
        pipe, search_space,
        scoring = 'accuracy',
        cv = 5, n_jobs = -1)
    best_clf.fit(X_train, y_train)
    elapsed = default_timer() - start

    # Mostramos los resultados
    print("--- Mejor clasificador lineal ---")
    print("Parámetros:\n{}".format(best_clf.best_params_['clf']))
    print("Número de variables usadas: {}".format(
        best_clf.best_estimator_['clf'].coef_.shape[1]))
    print("Accuracy en CV: {:.3f}%".format(100.0 * best_clf.best_score_))
    print_evaluation_metrics(
        best_clf,
        [X_train, X_test],
        [y_train, y_test],
        ["training", "test"])
    print("Tiempo: {:.3f}s".format(elapsed))

    # Gráficas y visualización
    if SHOW > 0:
        vs.wait()
        print("Mostrando gráficas sobre entrenamiento y predicción...")

        # Matriz de confusión
        vs.confusion_matrix(best_clf, X_test, y_test, SAVE_FIGURES, IMG_PATH)

        # Visualización de componentes principales
        if selection_strategy == Selection.PCA:
            # Predicciones para el conjunto de test
            y_pred = best_clf.predict(X_test)

            # Proyección de las dos primeras componentes principales
            # con etiquetas predichas
            vs.scatter_pca(X_test_pre, y_pred, SAVE_FIGURES, IMG_PATH)

            # Seleccionamos dos clases concretas y mostramos también los clasificadores,
            # frente a las etiquetas reales
            vs.scatter_pca_classes(
                X_test_pre, y_test,
                [best_clf.best_estimator_['clf'].coef_],
                ["Mejor clasificador lineal"],
                SAVE_FIGURES, IMG_PATH)

        if SHOW > 1:
            # Curva de aprendizaje
            print("Calculando curva de aprendizaje... ")
            start = default_timer()
            vs.plot_learning_curve(
                best_clf,
                X_train, y_train,
                n_jobs = -1, cv = 5,
                scoring = 'accuracy',
                save_figures = SAVE_FIGURES,
                img_path = IMG_PATH)
            elapsed = default_timer() - start
            print("Tiempo: {:.3f}s".format(elapsed))

def fit_dummy(X_train, X_test, y_train, y_test):
    """Ajustamos un clasificador que estima una clase aleatoria teniendo en
       cuenta la distribución de clases."""

    # Elegimos un clasificador aleatorio
    dummy_clf = DummyClassifier(strategy = 'stratified')

    # Ajustamos el modelo
    print("Ajustando un modelo dummy...\n")
    start = default_timer()
    dummy_clf.fit(X_train, y_train)
    elapsed = default_timer() - start

    # Mostramos los resultados
    print("--- Clasificador aleatorio ---")
    print("Número de variables usadas: {}".format(X_train.shape[1]))
    print_evaluation_metrics(
        dummy_clf,
        [X_train, X_test],
        [y_train, y_test],
        ["training", "test"])
    print("Tiempo: {:.3f}s".format(elapsed))

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

    print("----- PROYECTO FINAL: AJUSTE DE MODELOS DE CLASIFICACIÓN -----")

    # Cargamos los datos de entrenamiento y test
    print("Cargando datos de entrenamiento y test... ", end = "", flush = True)
    X, y, attr_names = read_data(PATH + DATASET_NAME)
    X_train, X_val, X_test, y_train, y_val, y_test = \
        split_data(X, y, val_size = 0.0, test_size = 0.3)
    print("Hecho.")

    # Inspeccionamos los datos
    if SHOW > 0:
        print("Mostrando gráficas de inspección de los datos...")

        # Mostramos distribución de clases en training y test
        vs.plot_class_distribution(y_train, y_test, N_CLASSES, SAVE_FIGURES, IMG_PATH)

        # Visualizamos las variables más relevantes
        # TODO: elegir las más relevantes por criterio RF
        features = [44, 57]
        vs.plot_features(
            features, attr_names[features],
            X_train, y_train,
            SAVE_FIGURES, IMG_PATH)

        if SHOW > 1:
            # Visualizamos el conjunto en 2 dimensiones
            vs.plot_tsne(X_train, y_train, SAVE_FIGURES, IMG_PATH)

    # Ajustamos un modelo lineal
    fit_linear(
        X_train, X_test,
        y_train, y_test,
        selection_strategy = Selection.PCA)

    # Ajustamos un modelo dummy
    fit_dummy(X_train, X_test, y_train, y_test)

    # Comparamos el mejor modelo encontrado de distintas clases
    #compare(...)

    # Imprimimos tiempo total de ejecución
    elapsed = default_timer() - start
    print("\nTiempo total de ejecución: {:.3f}s".format(elapsed))

if __name__ == "__main__":
    main()
