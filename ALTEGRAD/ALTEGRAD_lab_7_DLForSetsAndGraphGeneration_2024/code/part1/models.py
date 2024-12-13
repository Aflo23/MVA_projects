"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""
import numpy as np 
import torch
import torch.nn as nn

class DeepSets(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(DeepSets, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # why 1 unit ? 
        self.tanh = nn.Tanh()

    def forward(self, x):
        
        ############## Task 3
    
        ##################
        # your code here #

        x_2 = self.embedding(x)
        h_2 = self.fc1(x_2)
        h_2_activation = self.tanh(h_2)
        aggregator_1 = torch.sum(h_2_activation, dim=1)
        x = self.fc2(aggregator_1)


        ##################
        
        return x.squeeze()


class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        
        ############## Task 4
    
        ##################
        # your code here #
        x_2 = self.embedding(x)



        h_2_hidden, (h_n, c_n) = self.lstm(x_2)  # Get the output and hidden states from the LSTM
        

        #print(f'forward LSTM loop debugging {type(h_2_hidden[:,-1,:])} ')
        print(f"Type of h_2_hidden: {type(h_2_hidden)}")  # Should be a Tensor
        print(f"Shape of h_2_hidden: {h_2_hidden.shape}")  # Should be (batch_size, sequence_length, hidden_dim)
        # Extract the last time step's output (the last row of h_2_hidden)

        x = self.fc(h_2_hidden[:, -1, :])  # Use the output of the last time step
        
        ##################
        
        return x.squeeze()
