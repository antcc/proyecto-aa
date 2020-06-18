# Ajuste de modelos de clasificación en la base de datos [Online News Popularity](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)

Realizado junto a [@MiguelLentisco](https://github.com/MiguelLentisco). Curso 2019-20.

## Dependencias

### Código

- Python 3.7.7
- Scikit-Learn 0.23.1+
- NumPy 1.18.0+
- Pandas 1.0.4+
- Matplotlib 3.2.1+
- Seaborn 0.10.1+

### Documentación

- Pandoc 2.9.2.1+
- Filtros de Pandoc: [pandoc-citeproc](https://github.com/jgm/pandoc-citeproc)

## Resultados

 Modelo             | Accuracy en *test* (%)  | AUC en *test* (%)
:------------------:|:-----------------------:|:-----------------:
Regresión logística | 65.55                   | 70.92
Random Forest       | 66.71                   | 72.60
Gradient Boosting   | 66.44                   | 72.84
MLP                 | 64.83                   | 70.25
KNN                 | 63.95                   | 68.80
RBF-Network         | 64.83                   | 70.14
