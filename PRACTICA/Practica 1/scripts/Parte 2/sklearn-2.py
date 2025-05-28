import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

# Cargar los datos
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X_full = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
Y = raw_df.values[1::2, 2]
print(raw_df.columns)

print(X_full.shape)
print(Y.shape)
# Se elige la variable mas dependiente de la salida
selector = SelectKBest(f_regression, k=1)
selector.fit(X_full, Y)
X = X_full[:, selector.get_support()]

# Modelos
regressor_lin = LinearRegression()
regressor_svr = SVR(kernel="rbf", C=1e1, epsilon=1)
regressor_rnf = RandomForestRegressor()

# Calcular el error cuadrático medio (MSE) y los puntajes para cada modelo
scores_lin = cross_val_score(
    regressor_lin, X, Y, scoring="neg_mean_squared_error", cv=10
)
mse_lin = -scores_lin.mean()

scores_svr = cross_val_score(
    regressor_svr, X, Y, scoring="neg_mean_squared_error", cv=10
)
mse_svr = -scores_svr.mean()

scores_rnf = cross_val_score(
    regressor_rnf, X, Y, scoring="neg_mean_squared_error", cv=10
)
mse_rnf = -scores_rnf.mean()

# Imprimir los resultados
print("MSE Linear Regression (LIN):", mse_lin)
print("MSE Support Vector Regression (SVR):", mse_svr)
print("MSE Random Forest (RNF):", mse_rnf)

# Comparar los modelos y determinar el mejor
best_model = min(
    [(mse_lin, "LIN"), (mse_svr, "SVR"), (mse_rnf, "RNF")], key=lambda x: x[0]
)
print(f"El mejor modelo es {best_model[1]} con un MSE de {best_model[0]}")

# Calcular las métricas utilizando todas las características
X_all = X_full

# Modelos con todas las características
regressor_lin_all = LinearRegression()
regressor_svr_all = SVR(kernel="rbf", C=1e1, epsilon=1)
regressor_rnf_all = RandomForestRegressor()

scores_lin_all = cross_val_score(
    regressor_lin_all, X_all, Y, scoring="neg_mean_squared_error", cv=10
)
mse_lin_all = -scores_lin_all.mean()

scores_svr_all = cross_val_score(
    regressor_svr_all, X_all, Y, scoring="neg_mean_squared_error", cv=10
)
mse_svr_all = -scores_svr_all.mean()

scores_rnf_all = cross_val_score(
    regressor_rnf_all, X_all, Y, scoring="neg_mean_squared_error", cv=10
)
mse_rnf_all = -scores_rnf_all.mean()

# Imprimir los resultados con todas las características
print("\nMSE Linear Regression (LIN) con todas las características:", mse_lin_all)
print("MSE Support Vector Regression (SVR) con todas las características:", mse_svr_all)
print("MSE Random Forest (RNF) con todas las características:", mse_rnf_all)

# Comparar los modelos con todas las características
best_model_all = min(
    [(mse_lin_all, "LIN"), (mse_svr_all, "SVR"), (mse_rnf_all, "RNF")],
    key=lambda x: x[0],
)
print(
    f"El mejor modelo con todas las características es {best_model_all[1]} con un MSE de {best_model_all[0]}"
)

# Realizar la prueba de Wilcoxon para comparar LIN y SVR
estadistico, pvalor = wilcoxon(scores_lin, scores_svr)

print("\nPrueba de Wilcoxon (LIN vs. SVR):")
print("Estadístico:", estadistico)
print("Valor p:", pvalor)
