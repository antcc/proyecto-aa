#!/usr/bin/env python
# coding: utf-8
# uso: ./fit_reg.py

"""
Aprendizaje Automático. Curso 2019/20.
Proyecto final: ajuste del mejor modelo de regresión.
Intentamos conseguir el mejor ajuste posible dentro de una clase
acotada de modelos para un problema de regresión.

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

from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso, SGDRegressor, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

import visualization_reg as vs

#
# PARÁMETROS Y DEFINICIONES GLOBALES
#

SEED = 2020
PATH = "../datos/"
DATASET_NAME = "OnlineNewsPopularity.csv"
SAVE_FIGURES = False
IMG_PATH = "../doc/img/"
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

def print_evaluation_metrics(reg, X_lst, y_lst, names):
    """Imprime la evaluación de resultados en varios conjuntos de un regresor.
      - X_lst: lista de matrices de características.
      - y_lst: lista de vectores de etiquetas.
      - names: lista de nombres de los conjuntos (training, test, ...)."""

    for name, X, y in zip(names, X_lst, y_lst):
        y_pred = reg.predict(X)
        print("RMSE en {}: {:.3f}".format(
            name, np.sqrt(mean_squared_error(y, y_pred))))
        print("R2 en {}: {:.3f}".format(
            name, r2_score(y, y_pred)))

#
# LECTURA Y MANIPULACIÓN DE DATOS
#

def read_data(filename):
    """Carga los datos de fichero, elimina los atributos no predictivos, y separa
       los predictores, la columna objetivo y el nombre de los atributos."""

    # Cargamos los datos quitando las dos primeras columnas
    df = read_csv(
        filename,
        sep = ', ',
        engine = 'python',
        header = 0,
        usecols = [i for i in range(2, 62)],
        index_col = False,
        dtype = np.float64)

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
        train_test_split(X, y, test_size = test_size, random_state = SEED)

    # Divisón para validación
    if val_size > 0.0:
        X_train, X_val, y_train, y_val = \
            train_test_split(
                X_train, y_train,
                test_size = val_size / (1.0 - test_size),
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

#
# AJUSTE DE MODELOS
#

def fit_linear(X_train, X_test, y_train, y_test, selection_strategy = Selection.PCA):
    """Ajuste de un modelo lineal para un conjunto de datos."""

    # Creamos un pipeline de preprocesado
    preproc = preprocess_pipeline(Model.LINEAR, selection_strategy)
    preproc_pipe = Pipeline(preproc)

    # Obtenemos los datos preprocesados por si los necesitamos
    X_train_pre = preproc_pipe.fit_transform(X_train, y_train)
    X_test_pre = preproc_pipe.transform(X_test)

    # Construimos un pipeline de preprocesado + regresión
    # (el regresor que ponemos puede ser cualquiera, es un 'placeholder')
    pipe = Pipeline(preproc + [("reg", LinearRegression())])

    if SHOW > 0:
        print("Mostrando gráficas sobre preprocesado y características...")

        # Mostramos matriz de correlación de training antes y después de preprocesado
        vs.plot_corr_matrix(X_train, X_train_pre, SAVE_FIGURES, IMG_PATH)

    # Elegimos los modelos lineales y sus parámetros para CV
    max_iter = 1000
    search_space = [
        {"reg": [SGDRegressor(penalty = 'l2',
                              max_iter = max_iter,
                              random_state = SEED)],
         "reg__alpha": np.logspace(-4, 4, 5)},
        {"reg": [Ridge(max_iter = max_iter)],
         "reg__alpha": np.logspace(-4, 4, 5)},
        {"reg": [Lasso(random_state = SEED,
                       max_iter = max_iter)],
         "reg__alpha": np.logspace(-4, 4, 5)}]

    # Buscamos los mejores parámetros por CV
    print("Ajustando un modelo lineal...\n")
    start = default_timer()
    best_reg = GridSearchCV(
        pipe, search_space,
        scoring = 'neg_mean_squared_error',
        cv = 5, n_jobs = -1)
    best_reg.fit(X_train, y_train)
    elapsed = default_timer() - start

    # Mostramos los resultados
    print("--- Mejor regresor lineal ---")
    print("Parámetros:\n{}".format(best_reg.best_params_['reg']))
    print("Número de variables usadas: {}".format(
        best_reg.best_estimator_['reg'].coef_.shape[0]))
    print("RMSE en CV: {:.3f}".format(np.sqrt(-best_reg.best_score_)))
    print_evaluation_metrics(
        best_reg,
        [X_train, X_test],
        [y_train, y_test],
        ["training", "test"])
    print("Tiempo: {:.3f}s".format(elapsed))

    # Gráficas y visualización
    if SHOW > 0:
        vs.wait()
        print("Mostrando gráficas sobre entrenamiento y predicción...")

        # Visualización de residuos y error de predicción
        y_pred = best_reg.predict(X_test)
        vs.plot_residues_error(y_test, y_pred, SAVE_FIGURES, IMG_PATH)

        if selection_strategy == Selection.PCA:
            # Visualización de componentes principales
            m = best_reg.best_estimator_['reg'].coef_[0]
            b = best_reg.best_estimator_['reg'].intercept_
            vs.plot_scatter_pca_reg(X_test_pre[:, 0], y_test, m, b, SAVE_FIGURES, IMG_PATH)

        if SHOW > 1:
            # Curva de aprendizaje
            print("Calculando curva de aprendizaje...")
            vs.plot_learning_curve(
                best_reg,
                X_train, y_train,
                n_jobs = -1, cv = 5,
                scoring = 'neg_mean_squared_error',
                save_figures = SAVE_FIGURES,
                img_path = IMG_PATH)
            elapsed = default_timer() - start
            print("Tiempo: {:.3f}s".format(elapsed))

def fit_dummy(X_train, X_test, y_train, y_test):
    """Ajustamos un regresor que estima siempre la media de los datos."""

    # Elegimos un regresor dummy
    dummy_reg = DummyRegressor(strategy = 'mean')

    # Ajustamos el modelo
    print("Ajustando un modelo dummy...\n")
    start = default_timer()
    dummy_reg.fit(X_train, y_train)
    elapsed = default_timer() - start

    # Mostramos los resultados
    print("--- Regresor dummy ---")
    print("Número de variables usadas: {}".format(X_train.shape[1]))
    print_evaluation_metrics(
        dummy_reg,
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

    print("----- PROYECTO FINAL: AJUSTE DE MODELOS DE REGRESIÓN -----")

    # Cargamos los datos de entrenamiento y test
    print("Cargando datos de entrenamiento y test... ", end = "", flush = True)
    X, y, attr_names = read_data(PATH + DATASET_NAME)
    X_train, X_val, X_test, y_train, y_val, y_test = \
        split_data(X, y, val_size = 0.0, test_size = 0.3)
    print("Hecho.")

    # Inspeccionamos los datos
    if SHOW > 0:
        print("Mostrando gráficas de inspección de los datos...")

        # Visualizamos las variables más relevantes
        # TODO: elegir las más relevantes por criterio RF
        features = [44, 57]
        vs.plot_features(
            features, attr_names[features],
            X_train, y_train,
            SAVE_FIGURES, IMG_PATH)

        # Mostramos histograma de la variable dependiente
        vs.plot_hist_dependent(y_train, SAVE_FIGURES, IMG_PATH)

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
