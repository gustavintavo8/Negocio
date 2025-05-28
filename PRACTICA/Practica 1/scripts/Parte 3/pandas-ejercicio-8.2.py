import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def evaluar_modelos(df):
    # Dividir el conjunto de datos en características (X) y la variable objetivo (y)
    X = df[[
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
    ]]
    y = df["EstimatedSalary"]  # Simplificación: solo necesitamos la serie

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear un modelo de regresión lineal
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)

    # Crear un modelo de Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Calcular el MSE y R-squared para LinearRegression
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)

    # Calcular el MSE y R-squared para RandomForest
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"MSE Lin: {mse_linear}")
    print(f"R-squared Lin: {r2_linear}")
    print(f"MSE RNF: {mse_rf}")
    print(f"R-squared RNF: {r2_rf}")


# Método de imputación 1: eliminar filas con valores perdidos
print("****** Resultados con algoritmo 'Eliminación de filas con valores perdidos (dropna)' ******")
df = pd.read_excel("Churn_Modelling_NANs.xlsx", na_values="NA")
df = df.dropna(axis=0, how="any")
evaluar_modelos(df)

# Método de imputación 2: imputación por la mediana (numéricas) y moda (categóricas)
print("****** Resultados con algoritmo 'Imputación por la mediana(numéricas) y moda(categóricas)' ******")
df = pd.read_excel("Churn_Modelling_NANs.xlsx", na_values="NA")

# Imputación de mediana y moda
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

evaluar_modelos(df)

# Método de imputación 3: imputación por la media (numéricas) y moda (categóricas)
print("****** Resultados con algoritmo 'Imputación por la media(numéricas) y moda(categóricas)' ******")
df = pd.read_excel("Churn_Modelling_NANs.xlsx", na_values="NA")

# Imputación de media y moda
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

evaluar_modelos(df)

# Método de imputación 4: imputación por relleno de 0s (solo numéricos)
print("****** Resultados con algoritmo 'Rellenar valores perdidos(numéricos) con ceros' ******")
df = pd.read_excel("Churn_Modelling_NANs.xlsx", na_values="NA")

# Rellenar valores perdidos con ceros
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col].fillna(0, inplace=True)

evaluar_modelos(df)
