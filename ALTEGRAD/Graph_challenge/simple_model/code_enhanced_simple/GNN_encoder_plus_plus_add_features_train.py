import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import argparse
import csv
from tqdm import tqdm
from GNN_encoder_plus_plus_add_features import GNNAdjacencyPredictor
from utils import preprocess_dataset
print('hellooo B GOTCHAA')
# Argument parser
parser = argparse.ArgumentParser(description='Graph Adjacency Predictor')
parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate") #1e-4
parser.add_argument('--dropout', type=float, default=0.4, help="Dropout rate") #0.3
parser.add_argument('--batch-size', type=int, default=48, help="Batch size") #32
parser.add_argument('--n-max-nodes', type=int, default=50, help="Maximum number of nodes in graphs")
parser.add_argument('--num-epochs', type=int, default=310, help="Number of training epochs") #300
args = parser.parse_args()

# Model hyperparameters
input_dim = 7
hidden_dim_1 = 550 # 500
hidden_dim_2 = 500 #450
hidden_dim_3 = 328 #285
latent_dim = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocess datasets
trainset = preprocess_dataset("train", args.n_max_nodes, 14)
validset = preprocess_dataset("valid", args.n_max_nodes, 14)
testset = preprocess_dataset("test", args.n_max_nodes, 14)

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# Initialize model
model = GNNAdjacencyPredictor(
    input_dim=input_dim,
    hidden_dim_1=hidden_dim_1,
    hidden_dim_2=hidden_dim_2,
    hidden_dim_3=hidden_dim_3,
    latent_dim=latent_dim,
    n_max_nodes=args.n_max_nodes,
    dropout_rate=args.dropout
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# # Training function
# def train_gnn(model, train_loader, val_loader, optimizer, num_epochs, device):
#     best_val_loss = float('inf')

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0
#         for data in train_loader:
#             stats = data.stats.to(device)
#             adj_true = data.A.squeeze(1).to(device)

#             # Extract extra features during training
#             node_features = data.x.mean(dim=1).to(device) if hasattr(data, 'x') else None
#             adj_features = data.adj.sum(dim=1).to(device) if hasattr(data, 'adj') else None
#             extra_features = torch.cat([node_features, adj_features], dim=-1) if node_features is not None and adj_features is not None else None

#             optimizer.zero_grad()
#             adj_pred = model(stats, extra_features=extra_features)  # Pass extra features
#             loss = model.loss_function(adj_true, adj_pred)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#         train_loss /= len(train_loader)

#         # Validation phase
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for data in val_loader:
#                 stats = data.stats.to(device)
#                 adj_true = data.A.squeeze(1).to(device)

#                 # Use extra features in validation
#                 node_features = data.x.mean(dim=1).to(device) if hasattr(data, 'x') else None
#                 adj_features = data.adj.sum(dim=1).to(device) if hasattr(data, 'adj') else None
#                 extra_features = torch.cat([node_features, adj_features], dim=-1) if node_features is not None and adj_features is not None else None

#                 adj_pred = model(stats, extra_features=extra_features)  # Pass extra features
#                 val_loss += model.loss_function(adj_true, adj_pred).item()
#         val_loss /= len(val_loader)

#         print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#         # Save best model
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), "best_gnn_adjacency_predictor.pth")
# Training function
def train_gnn(model, train_loader, val_loader, optimizer, num_epochs, device):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            stats = data.stats.to(device)
            adj_true = data.A.squeeze(1).to(device)

            # Extract extra features during training
            node_features = data.x.mean(dim=1).to(device) if hasattr(data, 'x') else None
            adj_features = data.adj.sum(dim=1).to(device) if hasattr(data, 'adj') else None
            extra_features = torch.cat([node_features, adj_features], dim=-1) if node_features is not None and adj_features is not None else None

            optimizer.zero_grad()
            adj_pred = model(stats, extra_features=extra_features)  # Pass extra features
            loss = model.loss_function(adj_true, adj_pred)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                stats = data.stats.to(device)
                adj_true = data.A.squeeze(1).to(device)

                # Use extra features in validation
                node_features = data.x.mean(dim=1).to(device) if hasattr(data, 'x') else None
                adj_features = data.adj.sum(dim=1).to(device) if hasattr(data, 'adj') else None
                extra_features = torch.cat([node_features, adj_features], dim=-1) if node_features is not None and adj_features is not None else None

                adj_pred = model(stats, extra_features=extra_features)  # Pass extra features
                val_loss += model.loss_function(adj_true, adj_pred).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_gnn_adjacency_predictor.pth")

# Train the model
train_gnn(model, train_loader, val_loader, optimizer, args.num_epochs, device)

# Save predictions to CSV
output_file = "output_predictions_GNN_plus_added_features_3.csv"
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["graph_id", "edge_list"])

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Generating predictions"):
            stats = data.stats.to(device)

            # For testing, assume extra features are unavailable
            adj_pred = model(stats).sigmoid()  # Predict without extra features

            for i, graph_id in enumerate(data.filename):
                adj_matrix = adj_pred[i].cpu().numpy()
                threshold = 0.5
                adj_binary = (adj_matrix > threshold).astype(int)

                # Create edge list from adjacency matrix
                edges = [(u, v) for u in range(adj_binary.shape[0]) for v in range(adj_binary.shape[1]) if adj_binary[u, v] == 1]
                edge_list_text = ", ".join([f"({u}, {v})" for u, v in edges])

                writer.writerow([graph_id, edge_list_text])

print(f"Predictions saved to {output_file}")
