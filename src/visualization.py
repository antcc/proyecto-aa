# coding: utf-8

"""
Aprendizaje Automático. Curso 2019/20.
Proyecto final: ajuste del mejor modelo de clasificación.
Colección de funciones para visualización de gráficas para un
problema de clasificación.

Todas las funciones tienen parámetros 'save_figures' y 'img_path'
que permiten guardar las imágenes generadas en disco en vez de
mostrarlas.

Miguel Lentisco Ballesteros
Antonio Coín Castro
"""

#
# LIBRERÍAS
#

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.model_selection import learning_curve
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from sklearn.manifold import TSNE

#
# FUNCIONES DE VISUALIZACIÓN
#

def wait(save_figures = False):
    """Introduce una espera hasta que se pulse una tecla."""

    if not save_figures:
        input("(Pulsa [Enter] para continuar...)\n")
    plt.close()

def scatter_plot(X, y, axis, ws = None, labels = None, title = None,
        xlim = None, ylim = None, figname = "", cmap = cm.tab10,
        save_figures = False, img_path = ""):
    """Muestra un scatter plot de puntos (opcionalmente) etiquetados por clases,
       eventualmente junto a varias rectas de separación.
         - X: matriz de características de la forma [x1, x2].
         - y: vector de etiquetas o clases. Puede ser None.
         - axis: nombres de los ejes.
         - ws: lista de vectores 2-dimensionales que representan las rectas
           (se asumen centradas).
         - labels: etiquetas de las rectas.
         - title: título del plot.
         - xlim = [xmin, xmax]: límites del plot en el eje X.
         - ylim = [ymin, ymax]: límites del plot en el eje Y.
         - figname: nombre para guardar la gráfica en fichero.
         - cmap: mapa de colores."""

    # Establecemos tamaño, colores e información del plot
    plt.figure(figsize = (8, 6))
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    if title is not None:
        plt.title(title)

    # Establecemos los límites del plot
    if xlim is None:
        xlim = [np.min(X[:, 0]), np.max(X[:, 0])]
    if ylim is None:
        ylim = [np.min(X[:, 1]), np.max(X[:, 1])]
    scale_x = (xlim[1] - xlim[0]) * 0.01
    scale_y = (ylim[1] - ylim[0]) * 0.01
    plt.xlim(xlim[0] - scale_x, xlim[1] + scale_x)
    plt.ylim(ylim[0] - scale_y, ylim[1] + scale_y)

    # Mostramos scatter plot con leyenda
    scatter = plt.scatter(
        X[:, 0], X[:, 1],
        c = y, cmap = cmap,
        marker = '.')
    if y is not None:
        legend1 = plt.legend(
            *scatter.legend_elements(),
            title = "Clases",
            loc = "upper right")

    # Pintamos las rectas con leyenda
    if ws is not None:
        colors = cm.tab20b.colors

        for w, l, c in zip(ws, labels, colors):
            x = np.array([xlim[0] - scale_x, xlim[1] + scale_x])
            plt.plot(
                x, (-w[0] * x) / w[1],
                label = l, lw = 2, ls = "--",
                color = c)
        plt.legend(loc = "lower right")

    # Añadimos leyenda sobre las clases
    if y is not None:
        plt.gca().add_artist(legend1)

    if save_figures:
        plt.savefig(img_path + figname + ".png")
    else:
        plt.show()

    wait(save_figures)

def plot_corr_matrix(raw, preproc, title, save_title = "", save_figures = False, img_path = ""):
    """Muestra la matriz de correlación de un cierto conjunto, antes y
       después del preprocesado.
         - raw: datos antes del preprocesado.
         - preproc: datos tras el preprocesado.
         - title: título del plot.
         - save_title: título de la imagen."""

    fig, axs = plt.subplots(1, 2, figsize = (15, 6))
    fig.suptitle(title)

    # Correlación antes de preprocesar
    with np.errstate(invalid = 'ignore'):
        corr_matrix = np.abs(np.corrcoef(raw, rowvar = False))
    im = axs[0].matshow(corr_matrix, cmap = 'viridis')
    axs[0].title.set_text("Sin preprocesado")

    # Correlación tras preprocesado
    corr_matrix_post = np.abs(np.corrcoef(preproc, rowvar = False))
    axs[1].matshow(corr_matrix_post, cmap = 'viridis')
    axs[1].title.set_text("Con preprocesado")

    fig.colorbar(im, ax = axs.ravel().tolist(), shrink = 0.6)

    if save_figures:
        plt.savefig(img_path + "correlation_" + save_title + ".png")
    else:
        plt.show()
    wait(save_figures)

def plot_feature_importance(importances, n, pca, save_figures = False, img_path = ""):
    """Muestra las características más relevantes obtenidas según algún
       criterio, o bien las primeras componentes principales.
         - importances: vector de relevancia de características.
         - n: número de características a seleccionar.
         - pca: controla si se eligen las primeras componentes principales."""

    if pca:
        indices = range(0, n)
        title = "Importancia de componentes principales"
    else:
        indices = np.argsort(importances)[-n:][::-1]
        title = "Importancia de características"

    # Diagrama de barras para la relevancia
    plt.figure(figsize = (12, 6))
    plt.title(title)
    plt.xlabel("Índice")
    plt.ylabel("Importancia")
    plt.bar(range(n), importances[indices])
    plt.xticks(range(n), indices, fontsize = 8)
    if save_figures:
        plt.savefig(img_path + "importance.png")
    else:
        plt.show()
    wait(save_figures)

def plot_learning_curve(estimator, X, y, scoring, ylim = None, cv = None,
                        n_jobs = None, train_sizes = np.linspace(.1, 1.0, 5),
                        title = "", plot_fit_time = True,
                        plot_score_time = False, save_figures = False,
                        img_path = ""):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Adapted from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    scoring : A str (see model evaluation documentation) or a scorer callable
        object / function with signature scorer(estimator, X, y)

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    title : string
        Title of plot.

    plot_fit_time : boolean
        Whether to plot a graph of fit_time vs training examples.

    plot_score_time : boolean
        Whether to plot a graph of score vs fit_times.
    """

    fig, axes = plt.subplots(1, 3, figsize = (16, 6))
    #plt.suptitle(title, y = 0.96)

    if scoring == 'accuracy':
        score_name = "Accuracy"
    elif scoring == 'neg_mean_squared_error':
        score_name = "RMSE"
    else:
        score_name = scoring

    axes[0].set_title("Learning Curves")
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Número de ejemplos de entrenamiento")
    axes[0].set_ylabel(score_name)

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)

    if scoring == 'neg_mean_squared_error':
        train_scores = np.sqrt(-train_scores)
        test_scores = np.sqrt(-test_scores)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label=score_name + " en training")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label=score_name + " en cross-validation")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    if plot_fit_time:
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Número de ejemplos de entrenamiento")
        axes[1].set_ylabel("Tiempos de entrenamiento (s)")
        axes[1].set_title("Escalabilidad del modelo")

    # Plot fit_time vs score
    if plot_score_time:
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("Tiempos de entrenamiento (s)")
        axes[2].set_ylabel(score_name)
        axes[2].set_title("Desempeño del modelo")

    if save_figures:
        plt.savefig(img_path + "learning_curve_" + title + ".png")
    else:
        plt.show()
    wait(save_figures)

def plot_tsne(X, y, save_figures = False, img_path = ""):
    """Aplica el algoritmo TSNE para proyectar el conjunto X en 2 dimensiones,
       junto a las etiquetas correspondientes."""

    scatter_plot(
        TSNE().fit_transform(X), y,
        axis = ["x", "y"],
        title = "Proyección 2-dimensional con TSNE",
        figname = "tsne",
        save_figures = save_figures,
        img_path = img_path)

def scatter_pca(X, y_pred, save_figures = False, img_path = ""):
    """Proyección de las dos primeras componentes principales
       con sus etiquetas predichas.
         - X: matriz de características bidimensionales.
         - y_pred: etiquetas predichas."""

    scatter_plot(
        X[:, [0, 1]],
        y_pred,
        axis = ["Primera componente principal",
                "Segunda componente principal"],
        title = ("Proyección de las dos primeras componentes principales con "
            "etiquetas predichas"),
        figname = "scatter",
        save_figures = save_figures,
        img_path = img_path)

def confusion_matrix(clf, X, y, save_figures = False, img_path = ""):
    """Muestra la matriz de confusión de un clasificador en un conjunto de datos.
         - clf: clasificador.
         - X, y: conjunto de datos y etiquetas."""

    fig, ax = plt.subplots(1, 1, figsize = (8, 6))
    disp = plot_confusion_matrix(clf, X, y, cmap = cm.Blues, values_format = 'd', ax = ax)
    disp.ax_.set_title("Matriz de confusión")
    disp.ax_.set_xlabel("Etiqueta predicha")
    disp.ax_.set_ylabel("Etiqueta real")

    if save_figures:
        plt.savefig(img_path + "confusion_" + clf['clf'].__class__.__name__ + ".png")
    else:
        plt.show()
    wait(save_figures)

def plot_class_distribution(y_train, y_test, n_classes, save_figures = False, img_path = ""):
    """Muestra la distribución de clases en entrenamiento y test."""

    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    plt.suptitle("Distribución de clases", y = 0.96)

    # Diagrama de barras en entrenamiento
    unique, counts = np.unique(y_train.astype(int), return_counts = True)
    axs[0].bar(
        unique,
        counts,
        color = [cm.tab10.colors[0], cm.tab10.colors[-1]])
    axs[0].title.set_text("Entrenamiento")
    axs[0].set_xlabel("Clases")
    axs[0].set_ylabel("Número de ejemplos")
    axs[0].set_xticks([-1, 1])

    # Diagrama de barras en test
    unique, counts = np.unique(y_test.astype(int), return_counts = True)
    axs[1].bar(
        unique,
        counts,
        color = [cm.tab10.colors[0], cm.tab10.colors[-1]])
    axs[1].title.set_text("Test")
    axs[1].set_xlabel("Clases")
    axs[1].set_ylabel("Número de ejemplos")
    axs[1].set_xticks([-1, 1])

    if save_figures:
        plt.savefig(img_path + "class_distr.png")
    else:
        plt.show()
    wait(save_figures)

def plot_features(features, names, X, y,
        xlim = None, ylim = None, save_figures = False, img_path = ""):
    """Muestra la proyección de dos características con sus etiquetas correspondientes.
         - features: vector de índices de dos características.
         - names: nombres de las características.
         - X: matriz de todas características.
         - y: vector de etiquetas.
         - xlim = [xmin, xmax]: límites del plot en el eje X.
         - ylim = [ymin, ymax]: límites del plot en el eje Y.
         """

    scatter_plot(
        X[:, features], y,
        axis = names,
        title = "Proyección de las dos características más relevantes",
        xlim = xlim,
        ylim = ylim,
        figname = "scatter_relevance",
        save_figures = save_figures,
        img_path = img_path)

def plot_analysis(clf_cv, clf_name, hyps1, hyp_name1,
        hyps2 = None, hyp_name2 = None,
        x_logscale = False, test_time = False,
        save_figures = False, img_path = ""):
    """Muestra una evaluación de 1 o 2 hiperparámetros de un modelo
       durante cross-validation.
         - clf_cv: resultado de CV del modelo.
         - clf_name: nombre del clasificador.
         - hyps1, hyp_name1: primer hiperparámetro y su nombre.
         - hyps2, hyp_name2: segundo hiperparámetro y su nombre.
         - x_logscale: si se muestra escala logarítmica en el eje X.
         - test_time: si se mide el tiempo de evaluación en 'test'."""

    # Si tenemos dos hiperparametros
    two_hyp = hyps2 is not None
    # Resultados acc-cv y tiempo de entrenamiento
    cv_acc = np.array(clf_cv.cv_results_["mean_test_score"])
    cv_time = np.array(clf_cv.cv_results_["mean_fit_time"])
    # Para clf como knn, si tomar tiempo en test
    if test_time:
        cv_time = np.array(clf_cv.cv_results_["mean_score_time"])
    # Redimensionamos para dos parámetros
    if two_hyp:
        new_shape = (len(hyps1), len(hyps2))
        cv_acc = cv_acc.reshape(new_shape)
        cv_time = cv_time.reshape(new_shape)

    # Figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
    #plt.suptitle("Análisis de " + clf_name)

    # Si usar escala logarímica en el eje x
    if x_logscale:
        ax1.set_xscale("log")
        ax2.set_xscale("log")

    # cv-acc/time y heatmap
    if two_hyp:
        # Resultado cv-acc/time para hyp2 fijando hyp1
        for i in range(len(hyps1)):
            ax1.plot(hyps2, cv_acc[i], "-o", label = str(hyps1[i]))
            ax2.plot(hyps2, cv_time[i], "-o", label = str(hyps1[i]))
            # Nombre ejes
            ax1.set_xlabel(hyp_name2)
            ax1.set_ylabel("cv-acc")
            ax2.set_xlabel(hyp_name2)
            ax2.set_ylabel("fit time (s)")
        # Leyenda
        ax1.legend(title = hyp_name1, ncol = 2)
        ax2.legend(title = hyp_name1, ncol = 2)
        fig.tight_layout()

        if save_figures:
            plt.savefig(img_path + clf_name + "_acc_time.png")
        else:
            plt.show()
        wait(save_figures)

        # Nuevo plot para mapa de calor
        plt.figure(figsize = (8, 6))
        plt.title(clf_name)

        # acc-cv para mapa de calor
        cv_acc = np.array(clf_cv.cv_results_["mean_test_score"])
        data_dic = {hyp_name1: np.repeat(hyps1, len(hyps2)),
                    hyp_name2: hyps2 * len(hyps1),
                    "cv_acc": cv_acc}

        # Transformación a dataframe
        df = pd.DataFrame(data = data_dic)
        df = pd.pivot_table(df, values = "cv_acc", index = [hyp_name1],
            columns = hyp_name2)

        # Mapa de calor
        sns.heatmap(df, linewidth = 0.5, cmap = "RdBu")

        if save_figures:
            plt.savefig(img_path + clf_name + "_heatmap.png")
        else:
            plt.show()
        wait(save_figures)

    # Para un parámetro solo acc-cv/time
    else:
        # cv-acc
        ax1.plot(hyps1, cv_acc, "-o")
        # time
        ax2.plot(hyps1, cv_time, "-or")
        # Nombre ejes
        ax1.set_xlabel(hyp_name1)
        ax1.set_ylabel("cv-acc")
        ax2.set_xlabel(hyp_name1)
        ax2.set_ylabel("fit time (s)")
        fig.tight_layout()

        if save_figures:
            plt.savefig(img_path + clf_name + "_acc_time.png")
        else:
            plt.show()
        wait(save_figures)

def plot_auc(clfs, X, y, save_figures = False, img_path = ""):
    """Muestra la curva ROC para varios clasificadores ya entrenados.
         - clfs: clasificadores.
         - X: características.
         - y: etiquetas."""

    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    for clf in clfs:
        name = clf['clf'].__class__.__name__
        plot_roc_curve(clf, X, y, name = name, ax = ax)

    if save_figures:
        plt.savefig(img_path + "auc.png")
    else:
        plt.show()
    wait(save_figures)
