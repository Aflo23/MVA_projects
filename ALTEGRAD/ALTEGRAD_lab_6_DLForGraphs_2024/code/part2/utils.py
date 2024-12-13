"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
import torch
import random 
from random import shuffle
from random import randint



def create_dataset():
    Gs = list()
    y = list()

    ############## Task 5
    
    ##################
    for i in range(50):
        n = randint(10,20)
        Gs.append(nx.fast_gnp_random_graph(n, p=0.2))
        y.append(0)
        Gs.append(nx.fast_gnp_random_graph(n, p=0.4))
        y.append(1)
    #shuffle Gs and y
    indices = list(range(100))
    shuffle(indices)
    for i in range(100):
        Gs[indices[i]] = Gs[i]
        y[indices[i]] = y[i]
    ###################
    return Gs, y


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
       
