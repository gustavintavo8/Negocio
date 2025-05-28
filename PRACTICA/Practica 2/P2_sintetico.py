import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, f_classif, chi2, RFECV
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing

# Parte 1 -> Generación del dataset sintético
print("# Generación del dataset sintético")
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

# Convertir a DataFrame para facilitar la manipulación
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
data = pd.DataFrame(X, columns=feature_names)
data['target'] = y

# Calculamos la varianza máxima
varMax = data.iloc[:, :-1].var().max()

# 1. Eliminación de variables con poca varianza utilizando VarianceThreshold
print("# 1. Eliminación de variables con poca varianza utilizando VarianceThreshold")
sel_variance = VarianceThreshold(0.1 * varMax)
data_reducido = sel_variance.fit_transform(data.iloc[:, :-1])
print("Columnas seleccionadas varianza (VarianceThreshold):", data.columns[:-1][sel_variance.get_support()])

# 2. Eliminación de variables utilizando SelectPercentile y f_classif
print("# 2. Eliminación de variables utilizando SelectPercentile y f_classif")
selector_f_classif = SelectPercentile(score_func=f_classif, percentile=10)
data_reducido_f_classif = selector_f_classif.fit_transform(data.iloc[:, :-1], data['target'])
selected_columns = data.columns[:-1][selector_f_classif.get_support()]
print("Columnas seleccionadas (SelectPercentile con f_classif):", selected_columns)

# 3. Eliminación de variables utilizando SelectKBest con f_classif
print("# 3. Eliminación de variables utilizando SelectKBest con f_classif")
selector_f_classif_kbest = SelectKBest(score_func=f_classif, k=2)
data_reducido_f_classif_kbest = selector_f_classif_kbest.fit_transform(data.iloc[:, :-1], data['target'])
selected_columns_kbest = data.columns[:-1][selector_f_classif_kbest.get_support()]
print("Columnas seleccionadas (f_classif KBest):", selected_columns_kbest)

# 4. Eliminación recursiva de variables con RFECV
print("# 4. Eliminación recursiva de variables con RFECV")
estimator = SVC(kernel="linear")
sel = RFECV(estimator, step=1, cv=5, n_jobs=-1)
data_reducido_rfecv = sel.fit(data.iloc[:, :-1], data['target'])
print("Columnas seleccionadas recursivamente:", data.columns[:-1][sel.get_support()])

# Parte 2 -> Eliminación de variables usando SelectFromModel
print("# 5. Eliminación de variables usando SelectFromModel")
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(data.iloc[:, :-1], data['target'])
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Variables ordenadas por importancia
print("Variables ordenadas por importancia:")
for f in range(data.shape[1] - 1):
    print(f"{f + 1}. variable {indices[f]} ({importances[indices[f]]:.6f})")

# Gráfico de las importancias de las variables
plt.figure()
plt.title("Importancia de las variables")
plt.bar(range(data.shape[1] - 1), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(data.shape[1] - 1), feature_names, rotation=90)
plt.xlim([-1, data.shape[1] - 1])
plt.show()

# Escalado de las características
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(data.iloc[:, :-1])

print("Proceso completado.")