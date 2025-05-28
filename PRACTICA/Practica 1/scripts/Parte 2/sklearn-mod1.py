import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd

# Cargar el conjunto de datos Boston Housing
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X_full = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
Y = raw_df.values[1::2, 2]

# Preparar las variables para el gráfico
orden = np.argsort(Y)
horizontal = np.arange(Y.shape[0])

# Entrenar los modelos y almacenar sus predicciones
regressor_lr = LinearRegression()
regressor_lr.fit(X_full, Y)
pred_lr = regressor_lr.predict(X_full)[orden]

regressor_svr = SVR(kernel='rbf', C=10, epsilon=1)
regressor_svr.fit(X_full, Y)
pred_svr = regressor_svr.predict(X_full)[orden]

regressor_rf = RandomForestRegressor()
regressor_rf.fit(X_full, Y)
pred_rf = regressor_rf.predict(X_full)[orden]

regressor_mlp = MLPRegressor()
regressor_mlp.fit(X_full, Y)
pred_mlp = regressor_mlp.predict(X_full)[orden]

# Gráfico de las predicciones superpuestas
plt.scatter(horizontal, Y[orden], color='black', label='Valor Real')
plt.plot(horizontal, pred_lr, color='blue', linewidth=2, label='Linear Regression')
plt.plot(horizontal, pred_svr, color='red', linewidth=2, label='SVR')
plt.plot(horizontal, pred_rf, color='green', linewidth=2, label='Random Forest')
plt.plot(horizontal, pred_mlp, color='purple', linewidth=2, label='MLP')

# Añadir leyenda y mostrar el gráfico
plt.legend()
plt.xlabel('Instancias (ordenadas por Y)')
plt.ylabel('Precio de la vivienda')
plt.title('Comparación de modelos de regresión')
plt.show()