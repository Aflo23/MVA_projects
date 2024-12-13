"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """GAT layer"""
    def __init__(self, n_feat, n_hidden, alpha=0.05):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(n_feat, n_hidden, bias=False)
        self.a = nn.Linear(2*n_hidden, 1)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        
        ############## Task 1
    
        ##################
        # your code here #

        W_x = self.fc(x)

        indices = adj.coalesce().indices()
        # separe the origin and the destination of the edge 
        h_i = W_x[indices[0,:]]
        h_j = W_x[indices[1,:]]

        concat_features = torch.cat([h_i, h_j], dim=1)

        h = self.leakyrelu(self.a(concat_features))

        ##################

        h = torch.exp(h.squeeze())
        unique = torch.unique(indices[0,:])
        t = torch.zeros(unique.size(0), device=x.device)
        h_sum = t.scatter_add(0, indices[0,:], h)
        h_norm = torch.gather(h_sum, 0, indices[0,:])
        alpha = torch.div(h, h_norm)
        adj_att = torch.sparse.FloatTensor(indices, alpha, torch.Size([x.size(0), x.size(0)])).to(x.device)
        
        ##################
        # your code here #
        out = torch.sparse.mm(adj_att, W_x)  
        ##################

        return out, alpha




class GNN(nn.Module):
    """GNN model"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()
        self.mp1 = GATLayer(nfeat, nhid)
        self.mp2 = GATLayer(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        
        ############## Tasks 2 and 4
    
        ##################
        # your code here #

        # first GAT layer 
        x, alpha_1 = self.mp1(x, adj) # Quick memo of Altegrad 4
        x = self.relu(x) # then a relu
        x = self.dropout(x) # and to regularize let's apply a dropout 

        # second GAT Layer 
        x, alpha_2 = self.mp2(x, adj)
        x = self.relu(x)
        ''' Remark:
        x = self.dropout(x) we decide not to add another dropout 
        --> I feel it's not necessary since one was already given in step 1
        Adding too much dropout could lead to see our calculations fade 
        we can leave this for now and if we see the data overfitting then we could add another layer
        '''


        # we then finish our layers with a last linear layer 
        """
        why we do this actually ? 
        well the way I see it is that it's kind of common at the end of different layers 
        to then connect the values (logits) that come out of the previous layers to the different classes
        In a way there could be a more possible logits than different classes, so putting that linear layer 
        contributes to convert (categorize) thosd logits into the different class possibilities 

        There's a dimensionality reduction that a linear layer does. 
        In a way let's think it's like compressing the data in that sense that it projects all into a space of dimension the output class       
        
        Technically in lab 5 - we had 128 embeddings for our graphs and we have a class of n_class 
        so let's say we come up with 5 classes (n_class = 5) we ll have to bring 128 logits back to 5 classes
        """

        x =  self.fc(x)



        ##################

        return F.log_softmax(x, dim=1), alpha_2 # on our columns # we take the last attention score
    
