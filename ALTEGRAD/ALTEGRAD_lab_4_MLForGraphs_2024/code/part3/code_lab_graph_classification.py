"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

############## Task 7


#load Mutag dataset
def load_dataset():

    ##################
    # your code here #



    dataset = TUDataset(root="code/datasets/MUTAG", name="MUTAG")

    # Convert each graph in the dataset to a NetworkX graph
    Gs = [to_networkx(data, to_undirected=True) for data in dataset]
    ##################

    y = [data.y.item() for data in dataset]
    return Gs, y


Gs,y = load_dataset()

#Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 8
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(G_train), 4))
    
    ##################
    # your code here #
    # Sample n_samples size-3 graphlets from each graph in the training set
    for i, G in enumerate(Gs_train):
        graphlet_counts = np.zeros(4)
        
        for _ in range(n_samples):
          
            nodes = random.sample(G.nodes(), 3)
            subgraph = G.subgraph(nodes).to_undirected()  

            for j, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(subgraph, graphlet):
                    graphlet_counts[j] += 1
        

        phi_train[i] = graphlet_counts
    ##################

    phi_test = np.zeros((len(G_test), 4))
    
    ##################
    # your code here #
    # Sample n_samples size-3 graphlets from each graph in the test set
    for i, G in enumerate(Gs_test):
        graphlet_counts = np.zeros(4)
        
        for _ in range(n_samples):
            # Randomly sample 3 nodes
            nodes = random.sample(G.nodes(), 3)
            subgraph = G.subgraph(nodes).to_undirected()
            
            # Compare the subgraph to each graphlet
            for j, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(subgraph, graphlet):
                    graphlet_counts[j] += 1
        
        # Store the feature vector for the graph in the test set
        phi_test[i] = graphlet_counts

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


    ##################




K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)



############## Task 9

##################
# your code here #

K_train_gtk, K_test_gtk = graphlet_kernel(G_train, G_test, n_samples=200)

##################



############## Task 10

##################
# your code here #


clf_sp = SVC(kernel='precomputed')
clf_sp.fit(K_train_sp, y_train)  
y_pred_sp = clf_sp.predict(K_test_sp) 


clf_gtk = SVC(kernel='precomputed')
clf_gtk.fit(K_train_gtk, y_train) 
y_pred_gtk = clf_gtk.predict(K_test_gtk)  


accuracy_sp = accuracy_score(y_test, y_pred_sp)
accuracy_gtk = accuracy_score(y_test, y_pred_gtk)

print(f'Accuracy of SVM with Shortest Path Kernel: {accuracy_sp}')
print(f'Accuracy of SVM with Graphlet Kernel: {accuracy_gtk}')

##################
