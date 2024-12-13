"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import numpy as np
import torch

def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




def create_dataset():
    """Generates a dataset of graphs and their corresponding labels."""
    graphs = []
    labels = []
    features = []  # List to store features of each graph's nodes
    # Generate 50 graphs for each class
    for _ in range(50):
        # Class 0: G(n, 0.2) sparse graphs
        n = random.randint(10, 20)  # Randomly choose number of nodes
        G = nx.fast_gnp_random_graph(n, 0.2)  # Generate random graph
        graphs.append(G)  # Add graph to dataset
        labels.append(0)  # Class label is 0

        node_features = np.ones((n, 1))  # Here, we set all features to 1
        features.append(node_features)
        
        # Class 1: G(n, 0.4) dense graphs
        n = random.randint(10, 20)  # Randomly choose number of nodes
        G = nx.fast_gnp_random_graph(n, 0.4)  # Generate random graph
        graphs.append(G)  # Add graph to dataset
        labels.append(1)  # Class label is 1


        node_features = np.ones((n, 1))  # All nodes have the same feature
        features.append(node_features)

    return graphs, labels, features
