import seaborn as sns
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
# groceries = []
# for df in pd.read_csv('Grocery Products Purchase.csv', sep=',', header=None, chunksize=1, skiprows=[0]):
#     groceries.append(list(df.iloc[0].dropna()))

groceries = pd.read_csv("Grocery Products Purchase.csv")
transactions = groceries.map(str).values.tolist()
groceries = [[x for x in tr if x != 'nan'] for tr in transactions]

te = TransactionEncoder()
te_data = te.fit_transform(groceries)
groceries_encoded = pd.DataFrame(te_data,columns=te.columns_).astype(int)


# Find frequent itemsets
frequent_itemsets = apriori(groceries_encoded, min_support=0.025, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.25, num_itemsets=len(frequent_itemsets))

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to the graph
for _, rule in rules.iterrows():
    G.add_edge(', '.join(rule['antecedents']), ', '.join(rule['consequents']), weight=rule['lift'])

# Visualize the graph
plt.figure(figsize=(12, 6))
pos = nx.spring_layout(G, k=1.0, seed=22)
nx.draw(G, pos, with_labels=True, node_size=800, node_color='skyblue', font_size=10, edge_color='gray', width=[G[u][v]['weight'] for u, v in G.edges()])

plt.title("Association Rules from Groceries")
plt.show()
