import pandas as pd
cables_filename = 'cables.csv'
cables= pd.read_csv(cables_filename, sep=',', decimal='.')

#1. Limpiar el dataset, quitando los valores perdidos
print("Limpiando el dataset")
X =  cables.iloc[:, 1:]
y = cables.iloc[:,-1]
#Eliminamos la primera columna, ya que no es necesaria, son índices
cables = cables.iloc[:, 1:]
# Eliminamos las filas con valores perdidos (algún valor perdido)
cables = cables.dropna()
X = X.dropna()
print("cables:\n" , cables)
print("X:\n" , X)
print("\n\n\n\n_____")

#2. Escalar o normalizar las variables (Normalizamos o estandarizamos con minMax o StandardScaler)
print("\nEscalando / normalizando las variables (La y no se escala porque es la variable de salida)")
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(cables.iloc[:-1])
scalerX = preprocessing.StandardScaler().fit(X)
cables_norm = scaler.fit_transform(cables)
X_norm = scaler.fit_transform(X)
print("cables_norm:\n" , cables_norm)
print("X_norm:\n" , X_norm)

#3. Detectar las variables irrelevantes o redundantes
print("\nDetectando variables irrelevantes o redundantes")
from sklearn.feature_selection import SelectKBest, f_classif
sel_kbest = SelectKBest(f_classif, k=2)
cables_reduced = sel_kbest.fit_transform(cables_norm[:, :-1], cables_norm[:, -1])  #Asumiendo que la última columna es la variable objetivo
selected_columns = cables.columns[:-1][sel_kbest.get_support()]
print("\nColumnas seleccionadas por SelectKBest (f_classif):", selected_columns)

#4a.  Construir un modelo lineal
print("\nConstruyendo un modelo lineal")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
# Dividir los datos en características (X) y la variable objetivo (y)
X = cables_norm[:, :-1]  # Características (excluyendo la última columna)
y = cables_norm[:, -1]   # Variable objetivo (la última columna)

# Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de Regresión Lineal
modelo = LinearRegression()

# Entrenar el modelo en los datos de entrenamiento
modelo.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = modelo.predict(X_test)

# Evaluar el rendimiento del modelo
mse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n(Modelo lineal) Error Cuadrático Medio(modelo):", mse)
print("\n(Modelo lineal) R-cuadrado:", r2)

#4b. Construir un modelo con Random Forest
print("\nConstruyendo un modelo con Random Forest")
from sklearn.ensemble import RandomForestRegressor

# Crear un modelo de Random Forest
modelo_random_forest = RandomForestRegressor(n_estimators=100, random_state=0)

# Entrenar el modelo en los datos de entrenamiento
modelo_random_forest.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred_random_forest = modelo_random_forest.predict(X_test)

# Evaluar el rendimiento del modelo de Random Forest
mse_random_forest = root_mean_squared_error(y_test, y_pred_random_forest)
r2_random_forest = r2_score(y_test, y_pred_random_forest)

print("\n(Modelo Random Forest) Error Cuadrático Medio:", mse_random_forest)
print("\n(Modelo Random Forest) R-cuadrado:", r2_random_forest)

#5. Realizar la validación cruzada de ambos modelos y decidir cuál es la precisión del modelo conseguido
print("\nRealizando la validación cruzada")

from sklearn.model_selection import cross_val_score

# Realizar validación cruzada para el modelo lineal
cv_scores_lineal = cross_val_score(modelo, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse_lineal = (cv_scores_lineal * -1) ** 0.5  # Convertir de error cuadrático medio negativo a raíz del error cuadrático medio
cv_r2_lineal = cross_val_score(modelo, X, y, cv=5, scoring='r2')

# Realizar validación cruzada para el modelo de Random Forest
cv_scores_random_forest = cross_val_score(modelo_random_forest, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse_random_forest = (cv_scores_random_forest * -1) ** 0.5
cv_r2_random_forest = cross_val_score(modelo_random_forest, X, y, cv=5, scoring='r2')

# Calcular la precisión promedio de cada modelo
avg_rmse_lineal = cv_rmse_lineal.mean()
avg_r2_lineal = cv_r2_lineal.mean()
avg_rmse_random_forest = cv_rmse_random_forest.mean()
avg_r2_random_forest = cv_r2_random_forest.mean()

print("\nPrecisión de modelo lineal (RMSE promedio):", avg_rmse_lineal)
print("\nPrecisión de modelo lineal (R-cuadrado promedio):", avg_r2_lineal)
print("\nPrecisión de modelo Random Forest (RMSE promedio):", avg_rmse_random_forest)
print("\nPrecisión de modelo Random Forest (R-cuadrado promedio):", avg_r2_random_forest)

# Decidir cuál modelo tiene un mejor rendimiento
mejor_modelo = "Modelo Lineal" if avg_r2_lineal > avg_r2_random_forest else "Modelo Random Forest"
print("\nEl mejor modelo es:", mejor_modelo)

