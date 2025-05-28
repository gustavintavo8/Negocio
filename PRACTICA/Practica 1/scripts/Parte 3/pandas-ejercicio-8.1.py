import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

df = pd.read_excel("Churn_Modelling_NANs.xlsx", na_values='NA')

# Seguir desde aquí

# 1.Elimina las filas con valores perdidos
print ("-----Eliminando las filas con valores perdidos-----")
df = df.dropna(axis=0, how='any', inplace=False)
print(df)

# 2.Selecciona un dataframe 'XC' con las columnas
# CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
print ("-----Seleccionando un dataframe 'XC'-----")
XC = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
print(XC)

# 3.Selecciona un dataframe 'Y' con la columna 'Exited'
print ("-----Seleccionando un dataframe 'C'-----")
C = df['Exited']
print(C)

# 4.Haz tres clasificadores diferentes de C frente a 
#   XC y compara sus porcentajes de aciertos con validación cruzada (10 fold)

#Crear los modelos
classifier_log = LogisticRegression(max_iter=1000000)
classifier_svc = SVC(kernel='rbf')
classifier_ran = RandomForestClassifier()

# Evaluar los modelos con validación cruzada (10 fold)
print("-----Evaluacion de Modelos-----")
print ("-----Regresión Logistica-----")
scores_log = cross_val_score(classifier_log, XC, C.values.ravel(), cv=10, scoring='accuracy')
accuracy_log = scores_log.mean()
print("Accuracy Regression Logistica: ", accuracy_log)

print("-----Regresión SVC-----")
scores_svc = cross_val_score(classifier_svc, XC, C.values.ravel(), cv=10, scoring='accuracy')
accuracy_svc = scores_svc.mean()
print("Accuracy SVC: ", accuracy_svc)

print("-----Random Forest-----")
scores_ran = cross_val_score(classifier_ran, XC, C.values.ravel(), cv=10, scoring='accuracy')
accuracy_ran = scores_ran.mean()
print("Accuracy Random Forest: ", accuracy_ran)
