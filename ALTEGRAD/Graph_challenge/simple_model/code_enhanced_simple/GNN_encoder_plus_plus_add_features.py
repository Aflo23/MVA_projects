import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNAdjacencyPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, latent_dim, n_max_nodes, dropout_rate=0.3):
        super(GNNAdjacencyPredictor, self).__init__()
        print('INITIALIZATION GOOD')

        self.n_max_nodes = n_max_nodes

        # Encoder: Map `.stats` (and optionally extra features) to latent space
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.bn3 = nn.BatchNorm1d(hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, latent_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Decoder: Map latent space to adjacency matrix
        self.fc_decoder_1 = nn.Linear(latent_dim, hidden_dim_3)
        self.bn_decoder_1 = nn.BatchNorm1d(hidden_dim_3)
        self.fc_decoder_2 = nn.Linear(hidden_dim_3, hidden_dim_2)
        self.bn_decoder_2 = nn.BatchNorm1d(hidden_dim_2)
        self.fc_decoder_3 = nn.Linear(hidden_dim_2, hidden_dim_1)
        self.bn_decoder_3 = nn.BatchNorm1d(hidden_dim_1)
        self.fc_decoder_4 = nn.Linear(hidden_dim_1, n_max_nodes * n_max_nodes)

        # Residual alignment
        self.fc_decoder_residual = nn.Linear(latent_dim, hidden_dim_1)

        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, stats, extra_features=None):
        # If extra features are provided, concatenate them with `stats`
        if extra_features is not None:
            stats = torch.cat([stats, extra_features], dim=-1)

        # Encode `.stats` to latent space
        x = self.activation(self.bn1(self.fc1(stats)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        latent = self.fc4(x)

        # Decode latent representation to adjacency matrix
        x = self.activation(self.bn_decoder_1(self.fc_decoder_1(latent)))
        x = self.activation(self.bn_decoder_2(self.fc_decoder_2(x)))

        # Align latent space for residual connection if required
        if latent.size(1) != x.size(1):
            latent = self.fc_decoder_residual(latent)

        x = self.activation(self.bn_decoder_3(self.fc_decoder_3(x))) + latent  # Residual connection
        adj_pred = self.fc_decoder_4(x)
        adj_pred = adj_pred.view(-1, self.n_max_nodes, self.n_max_nodes)  # Reshape to adjacency matrix
        return adj_pred

    def loss_function(self, adj_true, adj_pred):
        # Reconstruction loss for adjacency matrix
        return F.binary_cross_entropy_with_logits(adj_pred, adj_true)





# class GNNAdjacencyPredictorEnhanced(nn.Module):
#     def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, latent_dim, n_max_nodes, extra_dim=0, dropout_rate=0.3):
#         super(GNNAdjacencyPredictorEnhanced, self).__init__()
#         print('INITIALIZATION ENHANCED')

#         self.n_max_nodes = n_max_nodes
#         self.extra_dim = extra_dim

#         # Encoder: Map `.stats` and optional features to latent space
#         self.fc1 = nn.Linear(input_dim + extra_dim, hidden_dim_1)
#         self.bn1 = nn.BatchNorm1d(hidden_dim_1)
#         self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
#         self.bn2 = nn.BatchNorm1d(hidden_dim_2)
#         self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
#         self.bn3 = nn.BatchNorm1d(hidden_dim_3)
#         self.fc4 = nn.Linear(hidden_dim_3, latent_dim)
#         self.dropout = nn.Dropout(p=dropout_rate)

#         # Decoder: Map latent space to adjacency matrix
#         self.fc_decoder_1 = nn.Linear(latent_dim, hidden_dim_3)
#         self.bn_decoder_1 = nn.BatchNorm1d(hidden_dim_3)
#         self.fc_decoder_2 = nn.Linear(hidden_dim_3, hidden_dim_2)
#         self.bn_decoder_2 = nn.BatchNorm1d(hidden_dim_2)
#         self.fc_decoder_3 = nn.Linear(hidden_dim_2, hidden_dim_1)
#         self.bn_decoder_3 = nn.BatchNorm1d(hidden_dim_1)
#         self.fc_decoder_4 = nn.Linear(hidden_dim_1, n_max_nodes * n_max_nodes)

#         # Residual alignment
#         self.fc_decoder_residual = nn.Linear(latent_dim, hidden_dim_1)

#         self.activation = nn.LeakyReLU(negative_slope=0.01)

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, stats, extra_features=None):
#         # Concatenate stats and additional features if available
#         if extra_features is not None:
#             stats = torch.cat([stats, extra_features], dim=-1)

#         # Encode to latent space
#         x = self.activation(self.bn1(self.fc1(stats)))
#         x = self.activation(self.bn2(self.fc2(x)))
#         x = self.activation(self.bn3(self.fc3(x)))
#         x = self.dropout(x)
#         latent = self.fc4(x)

#         # Decode latent representation to adjacency matrix
#         x = self.activation(self.bn_decoder_1(self.fc_decoder_1(latent)))
#         x = self.activation(self.bn_decoder_2(self.fc_decoder_2(x)))

#         # Align latent space for residual connection if required
#         if latent.size(1) != x.size(1):
#             latent = self.fc_decoder_residual(latent)

#         x = self.activation(self.bn_decoder_3(self.fc_decoder_3(x))) + latent  # Residual connection
#         adj_pred = self.fc_decoder_4(x)
#         adj_pred = adj_pred.view(-1, self.n_max_nodes, self.n_max_nodes)  # Reshape to adjacency matrix
#         return adj_pred

#     def loss_function(self, adj_true, adj_pred):
#         # Reconstruction loss for adjacency matrix
#         return F.binary_cross_entropy_with_logits(adj_pred, adj_true)
