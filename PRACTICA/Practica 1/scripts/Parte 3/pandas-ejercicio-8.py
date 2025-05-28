import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_excel("Churn_Modelling_NANs.xlsx", na_values='NA')

# Seguir desde aquí

# 1.Elimina las filas con valores perdidos
print ("-----Eliminando las filas con valores perdidos-----")
df = df.dropna(axis=0, how='any', inplace=False)
print(df)

# 2.Selecciona un dataframe 'X' con las columnas
# CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
print ("-----Seleccionando un dataframe 'X'-----")
X = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
print(X)

# 3.Selecciona un dataframe 'Y' con la columna 'EstimatedSalary'
print ("-----Seleccionando un dataframe 'Y'-----")
Y = df['EstimatedSalary']
print(Y)

# 4.Haz tres modelos diferentes (por ejemplo, regresión lineal, SVR y Random Forest) de Y 
#   frente a X y compara el error cuadrático medio de los tres con validación cruzada (10 
#   fold).
regressor_lr = LinearRegression()
regressor_svr = SVR(kernel='rbf')
regressor_rnf = RandomForestRegressor()

# Evaluar los modelos con validación cruzada (10 fold)
print("-----Evaluacion de Modelos-----")
print ("-----Regresión Lineal-----")
scores_lr = -cross_val_score(regressor_lr, X, Y.values.ravel(), cv=10, scoring='neg_mean_squared_error')
mse_lr = scores_lr.mean()
print("MSE Regresión Lineal: ", mse_lr)

print("-----Regresión SVR-----")
scores_svr = -cross_val_score(regressor_svr, X, Y.values.ravel(), cv=10, scoring='neg_mean_squared_error')
mse_svr = scores_svr.mean()
print("MSE Regresión SVR: ", mse_svr)

print("-----Random Forest-----")
scores_rnf = -cross_val_score(regressor_rnf, X, Y.values.ravel(), cv=10, scoring='neg_mean_squared_error')
mse_rnf = scores_rnf.mean()
print("MSE Random Forest: ", mse_rnf)
