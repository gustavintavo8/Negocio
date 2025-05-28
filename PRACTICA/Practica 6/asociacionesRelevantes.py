import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Cargar el conjunto de datos
groceries = pd.read_csv("Grocery Products Purchase.csv")
transactions = groceries.values.tolist()
groceries = [[x for x in tr if pd.notna(x)] for tr in transactions]

# Codificar los datos
te = TransactionEncoder()
te_data = te.fit_transform(groceries)
groceries_encoded = pd.DataFrame(te_data, columns=te.columns_).astype(bool)

# Definir rangos para los parámetros
min_support_values = [0.01, 0.02, 0.03]
confidence_values = [0.5, 0.6, 0.7]
lift_values = [1.0, 1.5, 2.0]

# Variables para almacenar las mejores reglas
best_rules = None
best_metric = 0

# Iterar sobre diferentes combinaciones de parámetros
for min_support in min_support_values:
    for confidence in confidence_values:
        for lift in lift_values:
            # Encontrar itemsets frecuentes
            frequent_itemsets = apriori(groceries_encoded, min_support=min_support, use_colnames=True)

            # Generar reglas de asociación (añadiendo el argumento num_itemsets)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift, num_itemsets=len(frequent_itemsets))

            # Filtrar reglas por confianza
            filtered_rules = rules[rules['confidence'] >= confidence]

            # Calcular una métrica combinada (puedes ajustar esto según tus objetivos)
            metric = len(filtered_rules) * confidence * lift

            # Actualizar las mejores reglas si la métrica actual es mejor
            if metric > best_metric:
                best_metric = metric
                best_rules = filtered_rules

# Mostrar las mejores reglas ordenadas por confianza
sorted_rules = best_rules.sort_values(by='confidence', ascending=False)
print(sorted_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])