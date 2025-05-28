import pandas as pd
import numpy as np

# Tutorial adaptado de
# https://towardsdatascience.com/a-complete-pandas-guide-2dc53c77a002

# Abandono de clientes de un banco

# Lectura del dataset completo
file_path1 = r"C:\Users\tavin\OneDrive - Universidad de Oviedo\4to AÑO\NEGOCIO\PRACTICA\Practica 1\scripts\Parte 3\Churn_Modelling.csv"
file_path2 = r"C:\Users\tavin\OneDrive - Universidad de Oviedo\4to AÑO\NEGOCIO\PRACTICA\Practica 1\scripts\Parte 3\Churn_Modelling_NANs.xlsx"
file_path3 = r"C:\Users\tavin\OneDrive - Universidad de Oviedo\4to AÑO\NEGOCIO\PRACTICA\Practica 1\scripts\Parte 3\miarchivo.xlsx"
df = pd.read_csv(file_path1)
df.head()

# Lectura de algunas columnas
# read_csv tiene opciones para indicar el separador, etc.
cols = ['CustomerId','CreditScore','NumOfProducts']
df = pd.read_csv(file_path1, usecols=cols)
df.head()

# Dimensiones
print(df.shape)

# Lectura de las primeras 500 filas
df = pd.read_csv(file_path1, usecols=cols, nrows=500)

print(df.shape)

# Lectura/escritura desde excel

df = pd.read_excel(file_path2)
print(df)

df.to_excel(file_path3)

# Creación de un dataframe a partir de un diccionario
df = pd.DataFrame({'a':np.random.rand(10),
                 'b':np.random.randint(10, size=10),
                 'c':[True,True,True,False,False,np.nan,np.nan,
                      False,True,True],
                 'b':['London','Paris','New York','Istanbul',
                      'Liverpool','Berlin',np.nan,'Madrid',
                      'Rome',np.nan],
                 'd':[3,4,5,1,5,2,2,np.nan,np.nan,0],
                 'e':[1,4,5,3,3,3,3,8,8,4]})
print(df)
