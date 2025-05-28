import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X_full = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
Y = raw_df.values[1::2, 2]

best_k = None
best_score = -float("inf")
k_values = list(range(1,14))

print("Probando k values desde 1 hasta 13")
for k in k_values:
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X_full, Y)

    model = LinearRegression()
    scores = cross_val_score(
        model, X_new, Y, cv=5, scoring="r2"
    ) 

    mean_score = scores.mean()
    print(f"K={str(k)} R2={str(mean_score)}")
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"El mejor valor de k es {best_k} con un score de {best_score}")
