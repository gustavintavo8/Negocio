#! Parte 1 -> Eliminación de variables con poca varianza
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif, SelectPercentile, chi2

iris_filename = "datasets-uci-iris.csv"
iris = pd.read_csv(
    iris_filename,
    sep=",",
    decimal=".",
    header=None,
    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "target"],
)

# Calculamos la varianza máxima
varMax = iris.iloc[:, 0:4].var().max()


#* 1. Eliminación de variables con poca varianza utilizando VarianceThreshold
print("# 1. Eliminación de variables con poca varianza utilizando VarianceThreshold")
# Obtenemos el selector de caracteristicas con el parámetro threshold el 10% de la var maxima
sel_variance = VarianceThreshold(0.1 * varMax)
# Aplicamos el selector a las columnas de las características
iris_reducido = sel_variance.fit_transform(iris.iloc[:, 0:4])
# Mostramos las columnas seleccionadas
print("Columnas seleccionadas varianza (VarianceThreshold):", iris.columns[:-1][sel_variance.get_support()])


#* 2. Eliminación de variables con poca varianza utilizando SelectPercentile
print("# 2. Eliminación de variables con poca varianza utilizando SelectPercentile")
# Definimos el selector de características utilizando SelectPercentile con el método chi2
# Puedes cambiar chi2 por otro método adecuado para tu conjunto de datos
selector = SelectPercentile(score_func=chi2, percentile=10)  # Cambia el valor de "percentile" según tus necesidades
# Aplicamos el selector a las columnas de las características
iris_reducido = selector.fit_transform(iris.iloc[:, 0:4], iris['target'])
# Mostramos las columnas seleccionadas
print("Columnas seleccionadas (SelectPercentile):", iris.columns[:-1][selector.get_support()])


#* 3. Eliminación de variables con poca varianza utilizando f_classif
print("# 3. Eliminación de variables con poca varianza utilizando f_classif")
# Definimos el selector de características utilizando f_classif
selector_f_classif = SelectPercentile(score_func=f_classif, percentile=10)  # Cambia el valor de "percentile" según tus necesidades
# Aplicamos el selector a las columnas de las características
iris_reducido_f_classif = selector_f_classif.fit_transform(iris.iloc[:, 0:4], iris['target'])
# Mostramos las columnas seleccionadas
print("Columnas seleccionadas (f_classif):", iris.columns[:-1][selector_f_classif.get_support()])


#* 4. Eliminación de variables con poca varianza utilizando SelectPercentile y f_classif
print("# 4. Eliminación de variables con poca varianza utilizando SelectPercentile y f_classif")
# Definimos el selector de características utilizando SelectPercentile con el método f_classif
selector_f_classif = SelectPercentile(score_func=f_classif, percentile=10)  # Cambia el valor de "percentile" según tus necesidades
# Aplicamos el selector a las columnas de las características
iris_reducido_f_classif = selector_f_classif.fit_transform(iris.iloc[:, 0:4], iris['target'])
# Mostramos las columnas seleccionadas
print("Columnas seleccionadas (SelectPercentile con f_classif):", iris.columns[:-1][selector_f_classif.get_support()])


#! Parte 2 -> Eliminación de variables basada en estadísticos univariantes ---------------------
print("\nEliminación de variables basada en estadísticos univariantes")

# <<<<<<<<<<<<<<<<<<< Iris Dataset >>>>>>>>>>>>>>>>>>>
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile

# <<<<<<<<<<<<<<<<<<< Kbest y chi2 >>>>>>>>>>>>>>>>>>>
sel_kbest = SelectKBest(chi2, k=2)
# Fijarnos en que en este caso se necesita en el método fit_transform
# el valor de la salida (target) que en este caso contiene la especie
# a la que pertenece la planta
iris_reducido1 = sel_kbest.fit_transform(iris.iloc[:, 0:4], iris.iloc[:, 4])
print("Columnas seleccionadas kbest chi2(iris dataset): ", iris.columns[:-1][sel_kbest.get_support()])

# <<<<<<<<<<<<<<<<<<< Kbest y f_clasif >>>>>>>>>>>>>>>>>>>
from sklearn.feature_selection import f_classif
sel_kbest = SelectKBest(f_classif, k=2)
iris_reducido1 = sel_kbest.fit_transform(iris.iloc[:, 0:4], iris.iloc[:, 4])
print("Columnas seleccionadas kbest f_classif(iris dataset): ", iris.columns[:-1][sel_kbest.get_support()])

# <<<<<<<<<<<<<<<<<<< SelectPercentile y chi2 >>>>>>>>>>>>>>>>>>>
from sklearn.feature_selection import SelectPercentile
sel_percentile = SelectPercentile(chi2, percentile=20)
iris_reducido2 = sel_percentile.fit_transform(iris.iloc[:, 0:4], iris.iloc[:, 4])
print("Columnas seleccionadas percentile chi2(iris dataset): ", iris.columns[:-1][sel_percentile.get_support()])

# <<<<<<<<<<<<<<<<<<< SelectPercentile y f_clasif >>>>>>>>>>>>>>>>>>>
sel_percentile = SelectPercentile(f_classif, percentile=20)
iris_reducido2 = sel_percentile.fit_transform(iris.iloc[:, 0:4], iris.iloc[:, 4])
print("Columnas seleccionadas percentile f_classif(iris dataset): ", iris.columns[:-1][sel_percentile.get_support()])

#! Parte 3 -> Eliminación recursiva de variables -----------------------------
print("\nEliminación recursiva de variables")
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
estimator = SVC(kernel="linear")
#Primer parametro es cuantas var se eliminan en cada iteracion, segundo cuantas particiones
sel = RFECV(estimator, step=1, cv=5)
iris_reducido2 = sel.fit(iris.iloc[:,0:4],iris.iloc[:,4])
print(sel.ranking_)
print(sel.support_)
print("Columnas seleccionadas recursivamente", iris.columns[0:4][sel.get_support()])

#! Parte 4 -> Eliminación de variables usando SelectFromModel ----------------
print("\nEliminación de variables usando SelectFromModel")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

# Clasificador basado en arboles de decision
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
# X, y son las variables de entrada y de salida del dataset
X = iris.iloc[:, 0:4]
y = iris.iloc[:, -1]
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
# Variables ordenadas por importancia
print("Variables ordenadas:")
nombresCaracteristicas= iris.columns[:-1]
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
