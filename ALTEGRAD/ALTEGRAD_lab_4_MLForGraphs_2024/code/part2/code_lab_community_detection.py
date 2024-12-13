"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
import random 
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
# Perform spectral clustering to partition graph G into k clusters



# def spectral_clustering(G, k):
#     adj_matrix = nx.adjacency_matrix(G)
 
#     degrees = np.array(adj_matrix.sum(axis=1)).flatten() 
#     D_inv_sqrt = sp.diags(np.power(degrees, -0.5)) 

#     I = sp.eye(adj_matrix.shape[0]) 
#     L_rw = I - D_inv_sqrt @ adj_matrix @ D_inv_sqrt  
#     eigvals, eigvecs = eigsh(L_rw, k=k, which='SM')


#     U = eigvecs

#     # let's then apply kmeans algorithm
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(U) 

#     clustering = {node: labels[node] for node in range(adj_matrix.shape[0])}
#     return clustering


def spectral_clustering(G, k):
    
    ##################
    # your code here #

    A = nx.to_numpy_array(G)
    
    degrees = np.sum(A, axis=1) 
    D = np.diag(degrees) 

    D_inv = np.diag(1.0 / degrees) 
    Lrw = np.eye(len(G)) - D_inv @ A  
    eigenvalues, eigenvectors = np.linalg.eig(Lrw)
    
 
    sorted_indices = np.argsort(eigenvalues)
    eigenvectors_sorted = eigenvectors[:, sorted_indices[:k]]  
    

    eigenvectors_sorted_real = np.real(eigenvectors_sorted)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(eigenvectors_sorted_real)  

    clustering = {node: kmeans.labels_[i] for i, node in enumerate(G.nodes())}
    
    return clustering
  ##################


############## Task 4

##################
# your code here #



file_path = "code/datasets/CA-HepTh.txt" 
G = nx.read_edgelist(file_path, delimiter='\t', comments='#', create_using=nx.Graph())

# extarct the connected graph and seek for the largest component
connected_components = list(nx.connected_components(G))
gcc_nodes = max(connected_components, key=len) 
G_gcc = G.subgraph(gcc_nodes)  

# apply spectral clustering 
k = 50 
clustering_result = spectral_clustering(G_gcc, k)

print(f"Number of nodes in GCC: {G_gcc.number_of_nodes()}")
print(f"Number of edges in GCC: {G_gcc.number_of_edges()}")
print(f"Clustering result: {clustering_result}")
##################




############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #

    modularity = 0
    m = G.number_of_edges()  
    clusters = set(clustering.values())  
    modularity_score = 0  

    for cluster in clusters:
        nodes_in_cluster = [node for node in clustering if clustering[node] == cluster]
        subgraph = G.subgraph(nodes_in_cluster)  
        
        l_c = subgraph.number_of_edges() 
        d_c = sum([G.degree(node) for node in nodes_in_cluster])  


        modularity += (l_c / m) - (d_c / (2 * m)) ** 2

    return modularity
    ##################
    




############## Task 6

##################
# your code here #

k = 50

spectral_clustering_result = spectral_clustering(G, k)
spectral_modularity = modularity(G, spectral_clustering_result)
print(f"Modularity (Spectral Clustering): {spectral_modularity:.4f}")

random_clustering_result = {node: random.randint(0, k - 1) for node in G.nodes()}
random_modularity = modularity(G, random_clustering_result)
print(f"Modularity (Random Clustering): {random_modularity:.4f}")
##################







