import matplotlib.pyplot as plt
from sklearn.svm import SVR
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
regressor_svr1 = SVR(kernel='linear', C=10, epsilon=1)
regressor_svr1.fit(X_full, Y)
pred_svr1 = regressor_svr1.predict(X_full)[orden]

regressor_svr2 = SVR(kernel='rbf', C=10, epsilon=1)
regressor_svr2.fit(X_full, Y)
pred_svr2 = regressor_svr2.predict(X_full)[orden]

regressor_svr3 = SVR(kernel='poly', C=10, epsilon=1)
regressor_svr3.fit(X_full, Y)
pred_svr3 = regressor_svr3.predict(X_full)[orden]

# Gráfico de las predicciones superpuestas
plt.scatter(horizontal, Y[orden], color='black', label='Valor Real')
plt.plot(horizontal, pred_svr1, color='red', linewidth=2, label='SVR1')
plt.plot(horizontal, pred_svr2, color='green', linewidth=2, label='SVR2')
plt.plot(horizontal, pred_svr3, color='pink', linewidth=2, label='SVR3')

# Añadir leyenda y mostrar el gráfico
plt.legend()
plt.xlabel('Instancias (ordenadas por Y)')
plt.ylabel('Precio de la vivienda')
plt.title('Comparación de modelos de regresión')
plt.show()