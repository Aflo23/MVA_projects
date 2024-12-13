# """
# Deep Learning on Graphs - ALTEGRAD - Nov 2024
# """

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class, device):
        super(GNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        
        ############## Task 6
    
        ##################
        eye = torch.eye(adj.size(0), device=adj.device).to_sparse()
        adj = adj + eye
        x = self.fc1(torch.sparse.mm(adj,x_in))
        x =  self.relu(x)
        x = self.fc2(torch.sparse.mm(adj,x))
        ##################
        
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(int(torch.max(idx).item())+1, x.size(1)).to(self.device)
        out = out.scatter_add_(0, idx, x) 
        
        ##################
        out = self.relu(out)
        out = self.fc4(out)
        ##################

        return F.log_softmax(out, dim=1)
    
