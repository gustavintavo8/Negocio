import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif, SelectPercentile, chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile

letter_filename = "datasets-uci-letter.csv"
letter = pd.read_csv(letter_filename, sep=",", decimal=".")

# print(letter.head)

# Calculamos la varianza máxima
varMax = letter.iloc[:, 0:16].var().max()

#* 1. Eliminación de variables con poca varianza utilizando VarianceThreshold
print("# 1. Eliminación de variables con poca varianza utilizando VarianceThreshold")
# Obtenemos el selector de caracteristicas con el parámetro threshold el 10% de la var maxima
sel_variance = VarianceThreshold(0.1 * varMax)
# aplicamos el selector a las columnas de las características
letter_reducido = sel_variance.fit_transform(letter.iloc[:, 0:16])
# Mostramos las columnas seleccionadas
print("Columnas seleccionadas varianza (VarianceThreshold):", letter.columns[:-1][sel_variance.get_support()])

#* 2. Eliminación de variables con poca varianza utilizando SelectPercentile y f_classif
print("# 2. Eliminación de variables con poca varianza utilizando SelectPercentile y f_classif")

# Definimos el selector de características utilizando SelectPercentile con el método f_classif
selector_f_classif = SelectPercentile(score_func=f_classif, percentile=10)  # Cambia el valor de "percentile" según tus necesidades

# Aplicamos el selector solo a las columnas de características, excluyendo la última columna
X = letter.iloc[:, :-1]  # Excluimos la última columna utilizando la notación de índices
letter_reducido_f_classif = selector_f_classif.fit_transform(X, letter.iloc[:, -1])

# Mostramos las columnas seleccionadas
selected_columns = letter.columns[:-1][selector_f_classif.get_support()]
print("Columnas seleccionadas (SelectPercentile con f_classif):", selected_columns)

#* 3. Eliminación de variables con poca varianza utilizando f_classif
print("# 3. Eliminación de variables con poca varianza utilizando f_classif")

# Definimos el selector de características utilizando f_classif
selector_f_classif = SelectKBest(score_func=f_classif, k=2)  # Cambia el valor de "k" según tus necesidades (número de características a retener)

# Aplicamos el selector solo a las columnas de características, excluyendo la última columna
X = letter.iloc[:, :-1]  # Excluimos la última columna utilizando la notación de índices
letter_reducido_f_classif = selector_f_classif.fit_transform(X, letter.iloc[:, -1])

# Mostramos las columnas seleccionadas
selected_columns = letter.columns[:-1][selector_f_classif.get_support()]
print("Columnas seleccionadas (f_classif):", selected_columns)

#* 4. Eliminación de variables con poca varianza utilizando SelectPercentile y f_classif
print("# 4. Eliminación de variables con poca varianza utilizando SelectPercentile y f_classif")

# Definimos el selector de características utilizando SelectPercentile con el método f_classif
selector_f_classif = SelectPercentile(score_func=f_classif, percentile=10)  # Cambia el valor de "percentile" según tus necesidades

# Aplicamos el selector solo a las columnas de características, excluyendo la última columna
X = letter.iloc[:, :-1]  # Excluimos la última columna utilizando la notación de índices
letter_reducido_f_classif = selector_f_classif.fit_transform(X, letter.iloc[:, -1])

# Mostramos las columnas seleccionadas
selected_columns = letter.columns[:-1][selector_f_classif.get_support()]
print("Columnas seleccionadas (SelectPercentile con f_classif):", selected_columns)



#! Parte 2 -> Eliminación de variables basada en estadísticos univariantes ---------------------

# <<<<<<<<<<<<<<<<<<< Kbest y chi2 >>>>>>>>>>>>>>>>>>>
sel_kbest = SelectKBest(chi2, k=2)
# Fijarnos en que en este caso se necesita en el método fit_transform
# el valor de la salida (target) que en este caso contiene la especie
# a la que pertenece la planta
letter_reducido = sel_kbest.fit_transform(letter.iloc[:, 0:16], letter.iloc[:, 16])
print(
    "Columnas seleccionadas kbest chi2(Letter dataset): ",
    letter.columns[:-1][sel_kbest.get_support()],
)

# <<<<<<<<<<<<<<<<<<< Kbest y f_clasif >>>>>>>>>>>>>>>>>>>
from sklearn.feature_selection import f_classif

sel_kbest = SelectKBest(f_classif, k=2)
letter_reducido = sel_kbest.fit_transform(letter.iloc[:, 0:16], letter.iloc[:, 16])
print(
    "Columnas seleccionadas kbest f_classif(letter dataset): ",
    letter.columns[:-1][sel_kbest.get_support()],
)

# <<<<<<<<<<<<<<<<<<< SelectPercentile y chi2 >>>>>>>>>>>>>>>>>>>
from sklearn.feature_selection import SelectPercentile

sel_percentile = SelectPercentile(chi2, percentile=20)
letter_reducido2 = sel_percentile.fit_transform(letter.iloc[:, 0:16], letter.iloc[:, 16])
print(
    "Columnas seleccionadas percentile chi2(letter dataset): ",
    letter.columns[:-1][sel_percentile.get_support()],
)

# <<<<<<<<<<<<<<<<<<< SelectPercentile y f_clasif >>>>>>>>>>>>>>>>>>>
sel_percentile = SelectPercentile(f_classif, percentile=20)
letter_reducido2 = sel_percentile.fit_transform(letter.iloc[:, 0:16], letter.iloc[:, 16])
print(
    "Columnas seleccionadas percentile f_classif(letter dataset): ",
    letter.columns[:-1][sel_percentile.get_support()],
)

#! Parte 3 -> Eliminación recursiva de variables -----------------------------
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC

# Establecer el clasificador SVC con kernel lineal
estimator = SVC(kernel="linear")

# Configurar RFECV para usar todos los núcleos disponibles
sel = RFECV(estimator, step=1, cv=5, n_jobs=-1)

# Realizar la eliminación recursiva de variables en el conjunto de datos "letter"
letter_reducido2 = sel.fit(letter.iloc[:, 0:16], letter.iloc[:, 16])

# Imprimir los resultados
print(sel.ranking_)
print(sel.support_)
print("Columnas seleccionadas recursivamente", letter.columns[0:16][sel.get_support()])

#! Parte 4 -> Eliminación de variables usando SelectFromModel ----------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

# Clasificador basado en arboles de decision
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
# X, y son las variables de entrada y de salida del dataset
X = letter.iloc[:, 0:16]
y = letter.iloc[:, -1]
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
# Variables ordenadas por importancia
print("Variables ordenadas:")
nombresCaracteristicas= letter.columns[:-1]
for f in range(X.shape[1]):
    print("%d. variable %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# Grafico con las importancias de las variables
plt.figure()
plt.title("Importancia de las variables")
plt.bar(
    range(X.shape[1]),
    importances[indices],
    color="r",
    yerr=std[indices],
    align="center",
)
plt.xticks(range(X.shape[1]), nombresCaracteristicas[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Nota: en el datasets sintético se obtienen valores positivos y negaticos en las características
# hay métodos de selecciḉon de características que no admiten valores negativos
# para solucionar esto podéis usar cualquier escalado, por ejemplo
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
# X o lo que sea
X = scaler.fit_transform(X)
