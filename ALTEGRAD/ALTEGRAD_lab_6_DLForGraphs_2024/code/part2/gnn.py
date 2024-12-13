"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import time
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim

from models import GNN
from utils import create_dataset, sparse_mx_to_torch_sparse_tensor
from scipy.sparse import block_diag

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("first time hello")
# Hyperparameters
epochs = 200
batch_size = 8
n_hidden_1 = 16
n_hidden_2 = 32
n_hidden_3 = 32
learning_rate = 0.01

# Generates synthetic dataset
Gs, y = create_dataset()
n_class = np.unique(y).size

# Splits the dataset into a training and a test set
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.1)

N_train = len(G_train)
N_test = len(G_test)

# Initializes model and optimizer
model = GNN(1, n_hidden_1, n_hidden_2, n_hidden_3, n_class, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# Trains the model
for epoch in range(epochs):
    t = time.time()
    model.train()
    
    train_loss = 0
    correct = 0
    count = 0
    for i in range(0, N_train, batch_size):
        adj_batch = list()
        idx_batch = list()
        y_batch = list()

        ############## Task 7
        
        ##################
        # your code here #


        graphs_in_batch = G_train[i:i+batch_size]
        
        labels_in_batch = y_train[i:i+batch_size]

        # node 
        features_batch = []


        for graph_index, G in enumerate(graphs_in_batch):
            adj_matrix = nx.adjacency_matrix(G)
            number_of_nodes = adj_matrix.shape[0]
            adj_batch.append(adj_matrix)


            # node to graph mapping 
            idx_batch.extend([graph_index]*number_of_nodes)
            # implement 1 on each labels 
            features_batch.append(np.ones((number_of_nodes,1)))

            y_batch.append(labels_in_batch[graph_index])

        adj_block = block_diag(adj_batch)

        # Combine node features into a single matrix

        features_batch = np.vstack(features_batch)

        # let's just do all of the conversions:
        adj_batch = sparse_mx_to_torch_sparse_tensor(adj_block).to(device)  # Sparse tensor
        features_batch = torch.tensor(features_batch, dtype=torch.float32).to(device) 
        idx_batch = torch.tensor(idx_batch, dtype=torch.long).to(device) 
        y_batch = torch.tensor(y_batch, dtype=torch.long).to(device) 

        ##################
        
        optimizer.zero_grad()
        output = model(features_batch, adj_batch, idx_batch)
        loss = loss_function(output, y_batch)
        train_loss += loss.item() * output.size(0)
        count += output.size(0)
        preds = output.max(1)[1].type_as(y_batch)
        correct += torch.sum(preds.eq(y_batch).double())
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(train_loss / count),
              'acc_train: {:.4f}'.format(correct / count),
              'time: {:.4f}s'.format(time.time() - t))
        
print('Optimization finished!')

# Evaluates the model
model.eval()
test_loss = 0
correct = 0
count = 0
for i in range(0, N_test, batch_size):
    adj_batch = list()
    idx_batch = list()
    y_batch = list()

    ############## Task 7
    
    ##################
    # your code here #

    for i in range(0, N_test, batch_size):
        adj_batch = list()
        idx_batch = list()
        y_batch = list()

        # Get the current batch of graphs and labels
        batch_graphs = G_test[i:i + batch_size]
        batch_labels = y_test[i:i + batch_size]

        node_offset = 0
        features_batch = []
        
        # Loop through each graph in the batch
        for graph_idx, G in enumerate(batch_graphs):
            adj_matrix = nx.adjacency_matrix(G)  # Get adjacency matrix of the graph
            num_nodes = adj_matrix.shape[0]
            
            # Append adjacency matrix and update node indices
            adj_batch.append(adj_matrix)
            
            # Create node-to-graph mapping
            idx_batch.extend([graph_idx] * num_nodes)
            
            # Set constant features for all nodes (e.g., 1 for each node)
            features_batch.append(np.ones((num_nodes, 1)))  # 1 feature per node
            
            # Append graph label
            y_batch.append(batch_labels[graph_idx])
        
        # Combine adjacency matrices into a block diagonal matrix
        adj_block = block_diag(adj_batch)
        
        # Combine node features into a single matrix
        features_batch = np.vstack(features_batch)
        
        # Convert to PyTorch tensors
        adj_batch = sparse_mx_to_torch_sparse_tensor(adj_block).to(device)  # Sparse tensor
        features_batch = torch.tensor(features_batch, dtype=torch.float32).to(device)  # Dense tensor
        idx_batch = torch.tensor(idx_batch, dtype=torch.long).to(device)  # Node-to-graph mapping
        y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)  # Graph labels
        
    ##################

    output = model(features_batch, adj_batch, idx_batch)
    loss = loss_function(output, y_batch)
    test_loss += loss.item() * output.size(0)
    count += output.size(0)
    preds = output.max(1)[1].type_as(y_batch)
    correct += torch.sum(preds.eq(y_batch).double())

print('loss_test: {:.4f}'.format(test_loss / count),
      'acc_test: {:.4f}'.format(correct / count),
      'time: {:.4f}s'.format(time.time() - t))
