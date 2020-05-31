# Dataset

URL: https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity#
This dataset summarizes a heterogeneous set of features about articles published by Mashable in a period of two years.
39797 instances with 61 attributes (Integer, Real).
Target: no. of shares in social networks (Integer)

# Modelos

Es necesario comparar UN modelo lineal y al menos DOS de los siguientes; en cada caso hay que encontrar el mejor posible dentro de su clase. Cada modelo extra sobre esa base proporciona 5 puntos como máximo.

- **Perceptron Multicapa**: Considerar clases de funciones definida por arquitecturas con 3 capas y un número de unidades por capa en el rango 50-100. Considerar el número de neuronas por capa como un hiperparámetro.

- **Máquina de Soporte de Vectores (SVM)**: se recomienda el núcleo RBF-Gaussiano o el polinomial. Encontrar el mejor valor para los parámetros libres hasta una precisión de 2 cifras (enteras o decimales).

- **Boosting**: Para regresión se recomiendan los árboles como regresores simples justificando el parámetro de aprendiaje.

- **Random Forest**: Usar como hiperparámetros los valores que por defecto se dan en teoría y experimentar para obtener el número de árboles adecuado.

- **Red de Funciones de base Radial**: Hay que fijar el valor K en el modelo final. Pero se deben evaluar distintos valores de K como criterio para la elección del K final.

# Pasos a seguir

1. Definición del problema a resolver y enfoque elegido.
2. Argumentos a favor de la elección de los modelos.
3. Codificación de los datos de entrada para hacerlos útiles a los algoritmos.
4. Valoración del interés de la variables para el problema y selección de un subconjunto (en su caso).
5. Normalización de las variables (en su caso)
6. Justificación de la función de pérdida usada.
7. Selección de las técnica (parámetrica) y valoración de la idoneidad de la misma frente a otras alternativas
8. Aplicación de la técnica especificando claramente que algoritmos se usan en la estimación de los parámetros, los hiperparámetros y el error de generalización.
9. Argumentar sobre la idoneidad de la función regularización usada (en su caso)
10. Valoración de los resultados ( gráficas, métricas de error, análisis de residuos, etc )
11. Justificar que se ha obtenido la mejor de las posibles soluciones con la técnica elegida y la muestra dada. Argumentar en términos de los errores de ajuste y generalización,

# Criterios para datos perdidos

- Cuando una muestra de datos tengan más del 10 % de sus datos perdidos puede eliminarla del conjunto de datos si no afecta al tamaño del conjunto de datos. Si es más del 20 % debe eliminarla.

- Los datos perdidos se sustituiran por la suma del valor medio de la variable más un valor aleatorio en el intervalo [−1,5σ, 1,5σ] siendo σ la desviación típica de la dicha variable.

# Consideraciones

- Justificar elección de funciones y de TODOS los parámetros (incluidos por defecto).
- Explicar con detalle el funcionamiento de pipelines y gridSearch.
- Pueden usarse técnicas de reducción de dimensionalidad, ej. PCA o Random Projection si se justifica que su uso mejora los resultados.
- El uso de resultados y enfoques existentes en la literatura sobre las bases de datos está permitido y de hecho se alienta, siempre y cuando se deja manifiestamente claro que uso se hace de dicha información/resultado y cual es la aportación del proyecto sobre la misma.
- Incluir las referencias de la bibliografía usada.
- Compilación con `make` en la carpeta `doc`. Compilación continua:
```
while true; do make; inotifywait -e modify,close_write memoria.md; done
```

# Dudas

- Hay que comparar todos los modelos que elijamos entre sí? Primero se elije el mejor dentro de su clase y luego se comparan todos?

# Enfoque

??

# Formato de la documentación

- Rellenar el `README.md`
- Rellenar anexo con funcionamiento del código? Listings?
- Paquetes de latex en `header.md`
- Números con dólares siempre.

# Formato del código

- Hacer `pylint fit.py` y `pylint visualization.py`.
- Indentación de 4 espacios.

# Enlaces de interés
