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

import visualization as vs

import numpy as np
from timeit import default_timer
import os
from enum import Enum
from joblib import Memory
from shutil import rmtree

from pandas import read_csv

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.fixes import loguniform
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier

from scipy.stats import randint

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
    NONE = 1

class Model(Enum):
    """Clases de modelos para los ajustes."""

    LINEAR = 0
    RF = 1
    BOOST = 2
    MLP = 3
    SIMILARITY = 4

class multi_randint():
    """Representa un entero aleatorio como una tupla (r, r, ..., r) de tamaño 'size',
       cogido sin reemplazamiento uniformemente en un intervalo [low, high]."""

    def __init__(self, low, high, size):
        self.low = low
        self.high = high
        self.size = size
        self.seen = []

    def rvs(self, random_state = 42):
        if len(self.seen) < self.high - self.low - 1:
            while True:
                sample = randint.rvs(
                    self.low, self.high, size = 1,
                    random_state = random_state)[0]

                if sample not in self.seen:
                    self.seen.append(sample)
                    return self.size * (sample,)

        return self.size * (0,)

class RBFNetworkClassifier(BaseEstimator, ClassifierMixin):
    """Implementación de un clasificador de red de funciones (gaussianas) de base radial.
       Internamente utiliza un clasificador lineal RidgeClassifier para ajustar
       los pesos del modelo final."""

    def __init__(self, k = 7, alpha = 1.0, batch_size = 100,
                 random_state = None):
        """Construye un clasificador con los parámetros necesarios:
             - k: número de centros a elegir.
             - alpha: valor de la constante regularización.
             - batch_size: tamaño del batch para el clustering no supervisado.
             - random_state: semilla aleatoria."""

        self.k = k
        self.alpha = alpha
        self.batch_size = batch_size
        self.random_state = random_state
        self.centers = None
        self.r = None

    def _choose_centers(self, X):
        """Usando k-means escoge los k centros de los datos."""

        init_size = 3 * self.k if 3 * self.batch_size <= self.k else None

        kmeans = MiniBatchKMeans(
            n_clusters = self.k,
            batch_size = self.batch_size,
            init_size = init_size,
            random_state = self.random_state)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

    def _choose_radius(self, X):
        """Escoge el radio para la transformación radial."""

        # "Diámetro" de los datos
        R = np.max(euclidean_distances(X, X))

        self.r = R / (self.k ** (1 / self.n_features_in_))

    def _transform_rbf(self, X):
        """Transforma los datos usando el kernel RBF."""

        return rbf_kernel(X, self.centers, 1 / (2 * self.r ** 2))

    def fit(self, X, y):
        """Entrena el modelo."""

        # Establecemos el modelo lineal subyacente
        self.model = RidgeClassifier(
            alpha = self.alpha,
            random_state = self.random_state)

        # Guardamos las clases y las características vistas durante el entrenamiento
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        # Obtenemos los k centros usando k-means
        self._choose_centers(X)

        # Elegimos el radio para el kernel RBF
        self._choose_radius(X)

        # Transformamos los datos usando kernel RBF respecto de los centros
        Z = self._transform_rbf(X)

        # Entrenamos el modelo lineal resultante
        self.model.fit(Z, y)

        # Guardamos los coeficientes obtenidos
        self.intercept_ = self.model.intercept_
        self.coef_ = self.model.coef_

        return self

    def score(self, X, y = None):
        # Transformamos datos con kernel RBF
        Z = self._transform_rbf(X)

        # Score del modelo lineal
        return self.model.score(Z, y)

    def predict(self, X):
        # Transformamos datos con kernel RBF
        Z = self._transform_rbf(X)

        # Predicciones del modelo lineal
        return self.model.predict(Z)

    def decision_function(self, X):
        # Transformamos datos con kernel RBF
        Z = self._transform_rbf(X)

        # Función de decisión del modelo lineal
        return self.model.decision_function(Z)

#
# PARÁMETROS GLOBALES
#

SEED = 2020
N_CLASSES = 2
CLASS_THRESHOLD = 1400
DO_MODEL_SELECTION = True
PATH = "../datos/"
DATASET_NAME = "OnlineNewsPopularity.csv"
CACHEDIR = "cachedir"
SHOW_ANALYSIS = True
SAVE_FIGURES = False
IMG_PATH = "../doc/img/"
SHOW = Show.NONE

#
# FUNCIONES AUXILIARES
#

def print_evaluation_metrics(clf, X_lst, y_lst, names):
    """Imprime la evaluación de resultados en varios conjuntos de unos clasificadores.
      - clf: lista de clasificadores como Pipelines.
      - X_lst: lista de matrices de características.
      - y_lst: lista de vectores de etiquetas.
      - names: lista de nombres de los conjuntos (training, test, ...)."""

    # Mostramos número de variables
    n_in = clf['clf'].n_features_in_
    if n_in is None:
        n_in = X_lst[0].shape[1]
    print("Número de variables usadas: {}".format(n_in))

    for name, X, y in zip(names, X_lst, y_lst):
        # Mostramos accuracy
        print("* Accuracy en {}: {:.3f}%".format(
            name, 100.0 * clf.score(X, y)))

        # Mostramos AUC
        if hasattr(clf, "predict_proba"):
            y_pred = clf.predict_proba(X)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_pred = clf.decision_function(X)
        else:
            continue
        print("* AUC en {}: {:.3f}%".format(
            name, 100.0 * roc_auc_score(y, y_pred)))

    print("")

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

def preprocess_pipeline(model_class, selection_strategy):
    """Construye una lista de transformaciones para el
       preprocesamiento de datos, incluyendo transformaciones polinómicas
       y selección de características. Hay diferentes estrategias para cada
       clase de modelos."""

    preproc = []

    if model_class == Model.LINEAR:
        if selection_strategy == Selection.PCA:
            preproc = [
                ("var", VarianceThreshold()),
                ("standardize", StandardScaler()),
                ("selection", PCA(0.95, random_state = SEED)),
                ("poly", PolynomialFeatures(2, include_bias = False)),
                ("standardize2", StandardScaler())]
        else:
            preproc = [
                ("var", VarianceThreshold()),
                ("standardize", StandardScaler()),
                ("poly", PolynomialFeatures(2, include_bias = False)),
                ("standardize2", StandardScaler())]

    else:
        preproc = [
            ("var", VarianceThreshold()),
            ("standardize", StandardScaler())]

    return preproc

def preprocess_graphs(X):
    """Muestra matrices de correlación antes y después de cada uno de los tipos
       de preprocesado empleados."""

    pipe1 = Pipeline(preprocess_pipeline(Model.LINEAR, Selection.PCA))
    title1 = "Correlaciones en entrenamiento con selección PCA y aumento polinómico"
    st1 = "pca_poly"
    pipe2 = Pipeline(preprocess_pipeline(Model.LINEAR, Selection.NONE))
    title2 = "Correlaciones en entrenamiento sin selección y con aumento polinómico"
    st2 = "poly"
    pipe3 = Pipeline(preprocess_pipeline(Model.RF, Selection.NONE))
    title3 = "Correlaciones en entrenamiento sin selección ni aumento"
    st3 = "none"

    for pipe, title, st in zip([pipe1, pipe2, pipe3], [title1, title2, title3], [st1, st2, st3]):
        X_pre = pipe.fit_transform(X)
        vs.plot_corr_matrix(X, X_pre, title, st, SAVE_FIGURES, IMG_PATH)

#
# AJUSTE Y SELECCIÓN DE MODELOS
#

def fit_cv(X_train, y_train, clfs,
        model_class, selection_strategy = Selection.NONE,
        randomized = False, cv_steps = 20,
        n_jobs = -1, show_cv = False):
    """Ajuste de varios modelos de clasificación para un conjunto de datos, eligiendo
       por validación cruzada el mejor de ellos dentro de una clase concreta.
         - X_train, y_train: datos de entrenamiento.
         - clfs: lista de diccionarios con los modelos concretos para ajustar.
         - model_class: clase de modelos a ajustar (ver 'Model').
         - selection_strategy: estrategia de selección de características (ver 'Selection').
         - randomized: si es True, la búsqueda en grid se produce de forma aleatorizada.
         - cv_steps: si la búsqueda es aleatorizada, indica el nº de combinaciones a probar.
         - n_jobs: número de hebras para ejecuciones en paralelo.
         - show_cv: controla si se muestra un ranking con los mejores resultados en CV."""

    # Construimos un pipeline de preprocesado + clasificación (placeholder) con caché
    preproc = preprocess_pipeline(model_class, selection_strategy)
    memory = Memory(location = CACHEDIR, verbose = 0)
    pipe = Pipeline(preproc + [("clf", DummyClassifier())], memory = memory)

    if show_cv:
        print("Comparando modelos por validación cruzada... ", end = "", flush = True)
        start = default_timer()

    # Buscamos los mejores modelos por CV
    if randomized:
        best_clf = RandomizedSearchCV(
            pipe, clfs,
            scoring = 'accuracy',
            n_iter = cv_steps,
            cv = 5, n_jobs = n_jobs,
            verbose = 0)
    else:
        best_clf = GridSearchCV(
            pipe, clfs,
            scoring = 'accuracy',
            cv = 5, n_jobs = n_jobs,
            verbose = 0)

    best_clf.fit(X_train, y_train)

    if show_cv:
        elapsed = default_timer() - start
        print("Hecho.")
        print("Tiempo para {} candidatos: {:.3f} min\n".format(
            len(best_clf.cv_results_['params']), elapsed / 60.0))

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
        print("* Accuracy en CV: {:.3f}%\n".format(100.0 * best_clf.best_score_))

    # Limpiamos la caché
    memory.clear(warn = False)

    return best_clf

def fit_model_selection(X_train, X_val, y_train, y_val):
    """Realiza selección de modelos entre una serie de candidatos, devolviendo
       una lista con los mejores modelos encontrados, entrenados sobre el conjunto
       completo de entrenamiento.
         - X_train, y_train: datos de entrenamiento.
         - X_val, y_val: datos de validación.

       Para algunos modelos se realiza un preanálisis para determinar el espacio
       de búsqueda de algunos parámetros críticos, utilizando para ello un
       pequeño conjunto de validación."""

    X_train_full = np.vstack((X_train, X_val))
    y_train_full = np.concatenate((y_train, y_val))
    best_clfs = []

    # Preanálisis de modelos de RBF
    if SHOW_ANALYSIS:
        ks = [5, 10, 25, 50, 100, 200, 300, 400]
        alphas = [0.0, 1e-10, 1e-5, 1e-3, 1e-1, 1.0, 10.0]
        clfs_rbf = [
            {"clf": [RBFNetworkClassifier(random_state = SEED)],
             "clf__k": ks,
             "clf__alpha": alphas}]

        print("-> PREANÁLISIS: valor de k y constante de regularización\n")
        best_clf_rbf = fit_cv(
            X_val, y_val,
            clfs = clfs_rbf,
            model_class = Model.SIMILARITY,
            n_jobs = -1,
            show_cv = False)

        print("Mostrando resultado de preanálisis para RBF...")
        vs.plot_analysis(best_clf_rbf, "RBF",
            alphas, "alpha", ks, "k", x_logscale = True, reverse_order = True,
            save_figures = SAVE_FIGURES,
            img_path = IMG_PATH)

    return best_clfs

    #
    # CLASIFICADOR LINEAL
    #

    print("--- AJUSTE DE MODELOS LINEALES ---\n")

    # Preanálisis de modelos lineales
    if SHOW_ANALYSIS:
        max_iter = 1000
        clfs_lin = [
            {"clf": [LogisticRegression(penalty = 'l2',
                                        random_state = SEED,
                                        max_iter = max_iter)],
             "clf__C": np.logspace(-5, 1, 40)},
            {"clf": [RidgeClassifier(random_state = SEED,
                                     max_iter = max_iter)],
             "clf__alpha": np.logspace(-5, 5, 40)},
            {"clf": [SGDClassifier(random_state = SEED,
                                   penalty = 'l2',
                                   max_iter = max_iter)],
             "clf__alpha": np.logspace(-6, 2, 40)}]

        print("-> PREANÁLISIS: constante de regularización\n")
        for clf in clfs_lin:
            best_clf_lin = fit_cv(
                X_val, y_val,
                clfs = [clf],
                model_class = Model.LINEAR,
                selection_strategy = Selection.PCA,
                show_cv = False)

            name = clf['clf'][0].__class__.__name__
            if name == "LogisticRegression":
                param_name = "C"
            else:
                param_name = "alpha"

            print("Mostrando resultado de preanálisis para {}...".format(name))
            vs.plot_analysis(best_clf_lin, name,
                clf["clf__" + param_name], param_name,
                x_logscale = True,
                save_figures = SAVE_FIGURES, img_path = IMG_PATH)

    # Escogemos modelos lineales
    max_iter = 1000
    clfs_lin = [
        {"clf": [LogisticRegression(penalty = 'l2',
                                    random_state = SEED,
                                    max_iter = max_iter)],
         "clf__C": np.logspace(-4, 0, 9)},
        {"clf": [RidgeClassifier(random_state = SEED,
                                 max_iter = max_iter)],
         "clf__alpha": np.logspace(0, 5, 9)},
        {"clf": [SGDClassifier(random_state = SEED,
                               penalty = 'l2',
                               max_iter = max_iter,
                               eta0 = 0.1)],
         "clf__learning_rate": ['optimal', 'invscaling', 'adaptive'],
         "clf__alpha": np.logspace(-4, 0, 7)}]

    # Ajustamos el mejor modelo
    print("-> AJUSTE\n")
    best_clf_lin = fit_cv(
        X_train, y_train,
        clfs = clfs_lin,
        model_class = Model.LINEAR,
        selection_strategy = Selection.PCA).best_estimator_

    # Reentrenamos en el conjunto de entrenamiento completo
    best_clf_lin.fit(X_train_full, y_train_full)
    best_clfs.append(best_clf_lin)

    #
    # CLASIFICADOR RANDOM FOREST
    #

    print("--- AJUSTE DE RANDOM FOREST ---\n")

    # Preanálisis de Random Forest
    if SHOW_ANALYSIS:
        n_est = [100, 200, 300, 400, 500, 600]
        max_depth = [5, 10, 15, 20, 30, 40, 58]
        clfs_rf = [
            {"clf": [RandomForestClassifier(random_state = SEED)],
             "clf__max_depth": max_depth,
             "clf__n_estimators": n_est}]

        print("-> PREANÁLISIS: número de árboles y profundidad máxima\n")
        best_clf_rf = fit_cv(
            X_val, y_val,
            clfs = clfs_rf,
            model_class = Model.RF,
            show_cv = False)

        print("Mostrando resultado de preanálisis para RandomForest...")
        vs.plot_analysis(best_clf_rf, "RandomForest",
            max_depth, "max_depth",
            n_est, "n_estimators",
            save_figures = SAVE_FIGURES, img_path = IMG_PATH)

    # Escogemos modelos de Random Forest
    clfs_rf = [
        {"clf": [RandomForestClassifier(random_state = SEED,
                                        max_depth = 20)],
         "clf__n_estimators": [400, 600],
         "clf__ccp_alpha": [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]}]

    # Ajustamos el mejor modelo
    print("-> AJUSTE\n")
    best_clf_rf = fit_cv(
        X_train, y_train,
        clfs = clfs_rf,
        model_class = Model.RF).best_estimator_

    # Reentrenamos en el conjunto de entrenamiento completo
    best_clf_rf.fit(X_train_full, y_train_full)
    best_clfs.append(best_clf_rf)

    #
    # CLASIFICADOR BOOSTING
    #

    print("--- AJUSTE DE MODELOS DE BOOSTING ---\n")

    # Preanálisis de modelos de boosting
    if SHOW_ANALYSIS:
        ds = [1, 2, 3, 4, 5]
        clfs_boost = [
            {"clf": [AdaBoostClassifier(random_state = SEED,
                                        base_estimator = DecisionTreeClassifier())],
             "clf__base_estimator__max_depth": ds,
             "clf__n_estimators": [100, 200, 300, 400, 500]},
            {"clf": [GradientBoostingClassifier(random_state = SEED)],
             "clf__max_depth": ds,
             "clf__n_estimators": [100, 200, 300, 400, 500]}]

        print("-> PREANÁLISIS: número de árboles y profundidad máxima\n")
        for clf in clfs_boost:
            best_clf_boost = fit_cv(
                X_val, y_val,
                clfs = [clf],
                model_class = Model.BOOST,
                show_cv = False)

            name = clf['clf'][0].__class__.__name__

            print("Mostrando resultado de preanálisis para {}...".format(name))
            vs.plot_analysis(best_clf_boost, name,
                ds, "max_depth",
                clf["clf__n_estimators"], "n_estimators",
                save_figures = SAVE_FIGURES, img_path = IMG_PATH)

    # Escogemos modelos de Boosting
    clfs_boost = [
        {"clf": [AdaBoostClassifier(random_state = SEED)],
         "clf__n_estimators": [175, 200, 225],
         "clf__learning_rate": [0.5, 1.0]},
        {"clf": [GradientBoostingClassifier(random_state = SEED,
                                            n_estimators = 100)],
         "clf__learning_rate": [0.05, 0.1, 1.0],
         "clf__subsample": [1.0, 0.75],
         "clf__max_depth": [4, 5]},
        {"clf": [GradientBoostingClassifier(random_state = SEED,
                                            n_estimators = 300)],
         "clf__learning_rate": [0.05, 0.1, 1.0],
         "clf__subsample": [1.0, 0.75],
         "clf__max_depth": [1, 2]}]

    # Ajustamos el mejor modelo
    print("-> AJUSTE\n")
    best_clf_boost = fit_cv(
        X_train, y_train,
        clfs = clfs_boost,
        model_class = Model.BOOST).best_estimator_

    # Reentrenamos en el conjunto de entrenamiento completo
    best_clf_boost.fit(X_train_full, y_train_full)
    best_clfs.append(best_clf_boost)

    #
    # CLASIFICADOR MLP
    #

    print("--- AJUSTE DE MLP ---\n")

    # Preanálisis de MLP
    if SHOW_ANALYSIS:
        clfs_mlp = [
            {"clf": [MLPClassifier(random_state = SEED,
                               learning_rate_init = 0.01,
                               solver = 'sgd',
                               max_iter = 300,
                               learning_rate = 'adaptive',
                               activation = 'relu',
                               tol = 1e-3)],
             "clf__hidden_layer_sizes": multi_randint(50, 101, 2)}]

        print("-> PREANÁLISIS: número de neuronas por capa\n")
        best_clf_mlp = fit_cv(
            X_val, y_val,
            clfs = clfs_mlp,
            model_class = Model.MLP,
            randomized = True,
            cv_steps = 20,
            show_cv = False)

        params = best_clf_mlp.cv_results_['params']
        layers = np.sort([d['clf__hidden_layer_sizes'][0] for d in params])

        print("Mostrando resultado de preanálisis para MLP...")
        vs.plot_analysis(best_clf_mlp, "MLP",
            layers, "hidden_layer_sizes",
            save_figures = SAVE_FIGURES, img_path = IMG_PATH)

    # Escogemos modelos de MLP
    clfs_mlp = [
        {"clf": [MLPClassifier(random_state = SEED,
                               learning_rate_init = 0.1,
                               solver = 'sgd',
                               max_iter = 300,
                               learning_rate = 'adaptive',
                               activation = 'relu',
                               tol = 1e-3)],
         "clf__hidden_layer_sizes": [(57, 57), (88, 88)],
         "clf__alpha": loguniform(1e-2, 1e2)}]

    # Ajustamos el mejor modelo eligiendo 10 candidatos de forma aleatoria
    print("-> AJUSTE\n")
    best_clf_mlp = fit_cv(
        X_train, y_train,
        clfs = clfs_mlp,
        model_class = Model.MLP,
        randomized = True,
        cv_steps = 10).best_estimator_

    # Reentrenamos en el conjunto de entrenamiento completo
    best_clf_mlp.fit(X_train_full, y_train_full)
    best_clfs.append(best_clf_mlp)

    #
    # CLASIFICADOR KNN
    #

    print("--- AJUSTE DE KNN ---\n")

    # Preanálisis de KNN
    if SHOW_ANALYSIS:
        ks = [1, 3, 5, 10, 20, 25, 30, 40, 50, 100, 200]
        clfs_knn = [
            {"clf": [KNeighborsClassifier()],
             "clf__n_neighbors": ks}]

        print("-> PREANÁLISIS: valor de k\n")
        best_clf_knn = fit_cv(
            X_val, y_val,
            clfs = clfs_knn,
            model_class = Model.SIMILARITY,
            show_cv = False)

        print("Mostrando resultado de preanálisis para KNN...")
        vs.plot_analysis(best_clf_knn, "KNN",
            ks, "k", test_time = True,
            save_figures = SAVE_FIGURES, img_path = IMG_PATH)

    # Escogemos modelos de KNN
    clfs_knn = [
        {"clf": [KNeighborsClassifier()],
         "clf__n_neighbors": [80, 100, 120, 150],
         "clf__weights": ['uniform', 'distance']}]

    # Ajustamos el mejor modelo
    print("-> AJUSTE\n")
    best_clf_knn = fit_cv(
        X_train, y_train,
        clfs = clfs_knn,
        model_class = Model.SIMILARITY).best_estimator_

    # Reentrenamos en el conjunto de entrenamiento completo
    best_clf_knn.fit(X_train_full, y_train_full)
    best_clfs.append(best_clf_knn)

    #
    # CLASIFICADOR RBF
    #

    print("--- AJUSTE DE REDES DE FUNCIONES DE BASE RADIAL ---\n")

    # Preanálisis de modelos de RBF
    if SHOW_ANALYSIS:
        ks = [5, 10, 25, 50, 100, 200, 300]
        alphas = [0.0, 1e-10, 1e-5, 1e-3, 1e-1, 1.0, 10.0]
        clfs_rbf = [
            {"clf": [RBFNetworkClassifier(random_state = SEED)],
             "clf__k": ks,
             "clf__alpha": alphas}]

        print("-> PREANÁLISIS: valor de k y constante de regularización\n")
        best_clf_rbf = fit_cv(
            X_val, y_val,
            clfs = clfs_rbf,
            model_class = Model.SIMILARITY,
            n_jobs = -1,
            show_cv = False)

        print("Mostrando resultado de preanálisis para RBF...")
        vs.plot_analysis(best_clf_rbf, "RBF",
            ks, "k", alphas, "alpha", x_logscale = True,
            save_figures = SAVE_FIGURES,
            img_path = IMG_PATH)

    clfs_rbf = [
        {"clf": [RBFNetworkClassifier(random_state = SEED)],
         "clf__k": [5, 10, 25, 100],
         "clf__alpha": [0.001, 0.1, 1.0]}]

    # Ajustamos el mejor modelo
    print("-> AJUSTE\n")
    best_clf_rbf = fit_cv(
        X_train, y_train,
        clfs = clfs_rbf,
        model_class = Model.SIMILARITY,
        n_jobs = 1).best_estimator_

    # Reentrenamos en el conjunto de entrenamiento completo
    best_clf_rbf.fit(X_train_full, y_train_full)
    best_clfs.append(best_clf_rbf)

    #
    # CLASIFICADOR ALEATORIO
    #

    # Ajustamos un clasificador aleatorio
    dummy_clf = Pipeline([("clf", DummyClassifier(strategy = 'stratified'))])
    dummy_clf.fit(X_train_full, y_train_full)
    best_clfs.append(dummy_clf)

    return best_clfs

def fit_models(X_train, y_train):
    """Ajusta una serie de modelos prefijados a unos datos de entrenamiento.
       Estos modelos son los que se consideran los mejores dentro de cada
       una de sus clases.
         - X_train, y_train: datos de entrenamiento."""

    clfs = []

    #
    # CLASIFICADOR LINEAL
    #

    print("--- AJUSTE DE MODELO LINEAL ---\n")

    preproc = preprocess_pipeline(Model.LINEAR, Selection.PCA)
    clf_lin = Pipeline(preproc
        + [("clf", LogisticRegression(random_state = SEED,
                                      penalty = 'l2',
                                      max_iter = 1000,
                                      C = 0.1))])

    print("Entrenando clasificador lineal... ", end = "", flush = True)
    start = default_timer()
    clf_lin.fit(X_train, y_train)
    elapsed = default_timer() - start
    print("Hecho.")
    print("Tiempo de entrenamiento: {:.3f} min\n".format(elapsed / 60.0))
    clfs.append(clf_lin)

    #
    # CLASIFICADOR RANDOM FOREST
    #

    print("--- AJUSTE DE RANDOM FOREST ---\n")

    preproc = preprocess_pipeline(Model.RF, Selection.NONE)
    clf_rf = Pipeline(preproc
        + [("clf", RandomForestClassifier(random_state = SEED,
                                          n_estimators = 600,
                                          max_depth = 20,
                                          n_jobs = -1))])

    print("Entrenando clasificador Random Forest... ", end = "", flush = True)
    start = default_timer()
    clf_rf.fit(X_train, y_train)
    elapsed = default_timer() - start
    print("Hecho.")
    print("Tiempo de entrenamiento: {:.3f} min\n".format(elapsed / 60.0))
    clfs.append(clf_rf)

    #
    # CLASIFICADOR BOOSTING
    #

    print("--- AJUSTE DE MODELO DE BOOSTING ---\n")

    preproc = preprocess_pipeline(Model.BOOST, Selection.NONE)
    clf_gb = Pipeline(preproc
        + [("clf", GradientBoostingClassifier(random_state = SEED,
                                              n_estimators = 100,
                                              learning_rate = 0.1,
                                              max_depth = 4,
                                              subsample = 0.75))])

    print("Entrenando clasificador Gradient Boosting... ", end = "", flush = True)
    start = default_timer()
    clf_gb.fit(X_train, y_train)
    elapsed = default_timer() - start
    print("Hecho.")
    print("Tiempo de entrenamiento: {:.3f} min\n".format(elapsed / 60.0))
    clfs.append(clf_gb)

    #
    # CLASIFICADOR MLP
    #

    print("--- AJUSTE DE MODELO MLP ---\n")

    preproc = preprocess_pipeline(Model.MLP, Selection.NONE)
    clf_mlp = Pipeline(preproc
        + [("clf", MLPClassifier(random_state = SEED,
                                 hidden_layer_sizes = (88, 88),
                                 learning_rate_init = 0.1,
                                 solver = 'sgd',
                                 learning_rate = 'adaptive',
                                 activation = 'relu',
                                 tol = 1e-3,
                                 alpha = 3.0))])

    print("Entrenando clasificador MLP... ", end = "", flush = True)
    start = default_timer()
    clf_mlp.fit(X_train, y_train)
    elapsed = default_timer() - start
    print("Hecho.")
    print("Tiempo de entrenamiento: {:.3f} min\n".format(elapsed / 60.0))
    clfs.append(clf_mlp)

    #
    # CLASIFICADOR KNN
    #

    print("--- AJUSTE DE MODELO KNN ---\n")

    preproc = preprocess_pipeline(Model.SIMILARITY, Selection.NONE)
    clf_knn = Pipeline(preproc
        + [("clf", KNeighborsClassifier(n_neighbors = 150,
                                        weights = 'distance',
                                        n_jobs = -1))])

    print("Entrenando clasificador KNN... ", end = "", flush = True)
    start = default_timer()
    clf_knn.fit(X_train, y_train)
    elapsed = default_timer() - start
    print("Hecho.")
    print("Tiempo de entrenamiento: {:.3f} min\n".format(elapsed / 60.0))
    clfs.append(clf_knn)

    #
    # CLASIFICADOR RBF
    #

    print("--- AJUSTE DE MODELO DE REDES DE FUNCIONES DE BASE RADIAL ---\n")

    preproc = preprocess_pipeline(Model.SIMILARITY, Selection.NONE)
    clf_rbf = Pipeline(preproc
        + [("clf", RBFNetworkClassifier(random_state = SEED,
                                        k = 250,
                                        alpha = 1e-10))])

    print("Entrenando clasificador RBF... ", end = "", flush = True)
    start = default_timer()
    clf_rbf.fit(X_train, y_train)
    elapsed = default_timer() - start
    print("Hecho.")
    print("Tiempo de entrenamiento: {:.3f} min\n".format(elapsed / 60.0))
    clfs.append(clf_rbf)

    #
    # CLASIFICADOR ALEATORIO
    #

    print("--- AJUSTE DE MODELO ALEATORIO ---\n")

    clf_dummy = Pipeline([("clf", DummyClassifier(strategy = 'stratified'))])

    print("Entrenando clasificador aleatorio... ", end = "", flush = True)
    start = default_timer()
    clf_dummy.fit(X_train, y_train)
    elapsed = default_timer() - start
    print("Hecho.")
    print("Tiempo de entrenamiento: {:.3f} min\n".format(elapsed / 60.0))
    clfs.append(clf_dummy)

    #
    # CURVAS DE APRENDIZAJE
    #

    if SHOW == Show.ALL:
        print("Calculando y mostrando curvas de aprendizaje...\n")
        for clf in clfs:
            name = clf['clf'].__class__.__name__
            print("Clasificador: {}".format(name))
            vs.plot_learning_curve(
                clf,
                X_train, y_train,
                n_jobs = -1, cv = 5,
                scoring = 'accuracy',
                title = name,
                save_figures = SAVE_FIGURES,
                img_path = IMG_PATH)

    return clfs

#
# COMPARACIÓN DE MODELOS
#

def compare(clfs, X_train, X_test, y_train, y_test):
    """Compara una serie de modelos ya entrenados en unos datos concretos,
       añadiendo a la comparación un clasificador aleatorio.
         - clfs: clasificadores a comparar.
         - X_train, y_train: datos de entrenamiento.
         - X_test, y_test: datos de test."""

    for clf in clfs:
        print("--> {} <--".format(clf['clf']))
        if clf['clf'].__class__.__name__ == "KNeighborsClassifier":
            print_evaluation_metrics(
                clf,
                [X_test],
                [y_test],
                ["test"])
        else:
            print_evaluation_metrics(
                clf,
                [X_train, X_test],
                [y_train, y_test],
                ["training", "test"])

        if SHOW != Show.NONE:
            # Matriz de confusión
            print("Mostrando matriz de confusión en test para "
                + clf['clf'].__class__.__name__ + "...")
            vs.confusion_matrix(clf, X_test, y_test, SAVE_FIGURES, IMG_PATH)

    if SHOW != Show.NONE:
        # Mostramos gráfica de AUC del mejor modelo (RandomForest) y
        # del peor modelo (KNN)
        print("Mostrando la curva ROC para el mejor y el peor...")
        vs.plot_auc([clfs[1], clfs[-3]], X_test, y_test, SAVE_FIGURES, IMG_PATH)

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Función principal. Ejecuta el proyecto paso a paso.

       NOTA: Por motivos de unificar el código, todos los clasificadores
             considerados son un Pipeline, cuyo último paso es el
             clasificador en sí, con nombre 'clf'."""

    # Inicio de medición de tiempo
    start = default_timer()

    # Ignorar warnings de convergencia
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

    # Semilla aleatoria para reproducibilidad
    np.random.seed(SEED)

    # Número de decimales fijo para salida de vectores
    np.set_printoptions(formatter = {'float': lambda x: "{:0.3f}".format(x)})

    print("------- PROYECTO FINAL: AJUSTE DE MODELOS DE CLASIFICACIÓN -------\n")

    #
    # LECTURA DE DATOS
    #

    # Cargamos los datos de entrenamiento, validación y test (división 50-20-30)
    print("Leyendo datos de " + DATASET_NAME + "... ", end = "", flush = True)
    X, y, attr_names = read_data(PATH + DATASET_NAME)
    X_train, X_val, X_test, y_train, y_val, y_test = \
        split_data(X, y, val_size = 0.2, test_size = 0.3)
    X_train_full = np.vstack((X_train, X_val))
    y_train_full = np.concatenate((y_train, y_val))
    print("Hecho.\n")

    #
    # INSPECCIÓN DE LOS DATOS
    #

    if SHOW != Show.NONE:
        print("--- VISUALIZACIÓN DE LOS DATOS ---\n")

        # Mostramos distribución de clases en training y test
        print("Mostrando gráfica de distribución de clases...")
        vs.plot_class_distribution(y_train_full, y_test, N_CLASSES, SAVE_FIGURES, IMG_PATH)

        # Visualizamos la importancia de las características según RF
        print("Mostrando gráfica de importancia de características...")
        pipe = Pipeline([("var", VarianceThreshold()), ("std", StandardScaler())])
        X_train_full_pre = pipe.fit_transform(X_train_full)
        rf = RandomForestClassifier(200, random_state = SEED, max_depth = 20, n_jobs = -1)
        rf.fit(X_train_full_pre, y_train_full)

        vs.plot_feature_importance(
            rf.feature_importances_,
            n = X_train_full_pre.shape[1],
            pca = False,
            save_figures = SAVE_FIGURES,
            img_path = IMG_PATH)

        # Mostramos gráficas de preprocesado
        print("Mostrando matrices de correlación antes y después de cada preprocesado...")
        preprocess_graphs(X_train_full)

        if SHOW == Show.ALL:
            # Visualizamos el conjunto de entrenamiento en 2 dimensiones
            print("Mostrando proyección del conjunto de entrenamiento en dos dimensiones...")
            vs.plot_tsne(X_train_full, y_train_full, SAVE_FIGURES, IMG_PATH)

    if DO_MODEL_SELECTION:
        clfs = fit_model_selection(X_train, X_val, y_train, y_val)
    else:
        clfs = fit_models(X_train_full, y_train_full)

    #
    # COMPARACIÓN DE MODELOS
    #

    print("--- COMPARACIÓN DE LOS MEJORES MODELOS ---\n")

    compare(clfs, X_train_full, X_test, y_train_full, y_test)

    #
    # ESTADÍSTICAS Y LIMPIEZA
    #

    # Imprimimos tiempo total de ejecución
    elapsed = default_timer() - start
    print("Tiempo total de ejecución: {:.3f} min".format(elapsed / 60.0))

    # Eliminamos directorio de caché
    if os.path.isdir(CACHEDIR):
        rmtree(CACHEDIR)

if __name__ == "__main__":
    main()
