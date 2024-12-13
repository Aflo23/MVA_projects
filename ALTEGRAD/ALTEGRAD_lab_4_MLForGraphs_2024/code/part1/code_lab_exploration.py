"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1


file_path = "code/datasets/CA-HepTh.txt"


G = nx.read_edgelist(
    file_path,
    delimiter='\t',
    comments='#',
    create_using=nx.Graph()  
)


num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")



############## Task 2



# Compute all connected components
connected_components = list(nx.connected_components(G))
num_components = len(connected_components)

# Print the number of connected components
print(f"Number of connected components: {num_components}")

# Extract the largest connected component
largest_cc = max(connected_components, key=len)  # Find the largest component by size
largest_cc_subgraph = G.subgraph(largest_cc)  # Create a subgraph for the largest component

# Compute the number of nodes and edges in the largest connected component
largest_cc_num_nodes = largest_cc_subgraph.number_of_nodes()
largest_cc_num_edges = largest_cc_subgraph.number_of_edges()

# Compute the fraction of nodes and edges in the largest connected component
node_fraction = largest_cc_num_nodes / num_nodes
edge_fraction = largest_cc_num_edges / num_edges

# Print the results
print(f"Largest connected component has {largest_cc_num_nodes} nodes and {largest_cc_num_edges} edges")
print(f"Fraction of nodes in the largest connected component: {node_fraction:.4f}")
print(f"Fraction of edges in the largest connected component: {edge_fraction:.4f}")

