import os
import math
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
import torch
import torch.nn.functional as F
import community as community_louvain

from torch import Tensor
from torch.utils.data import Dataset

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from tqdm import tqdm
import scipy.sparse as sparse
from torch_geometric.data import Data

from extract_feats import extract_feats, extract_numbers

print('ok ok')


# Original 
def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim):

    data_lst = []
    if dataset == 'test':
        filename = './data/dataset_'+dataset+'.pt' #'./data/dataset_'+dataset+'_processing_new.pt'
        desc_file = './data/'+dataset+'/test.txt'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)
                feats_stats = extract_numbers(desc)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
                data_lst.append(Data(stats=feats_stats, filename = graph_id))
            fr.close()                    
            torch.save(data_lst, filename)
            #print(f'Dataset {filename} saved')


    else:
        # filename = './data/dataset_'+dataset+'.pt'
        # graph_path = './data/'+dataset+'/graph'
        # desc_path = './data/'+dataset+'/description'
        filename = './data/dataset_'+dataset+'.pt'
        graph_path = './data/'+dataset+'/graph'
        desc_path = './data/'+dataset+'/description'

        if os.path.isfile(filename):
            " dejà loadé "
            data_lst = torch.load(filename)
            #print(f'Dataset {filename} loaded from file')

        else:
            # traverse through all the graphs of the folder
            files = [f for f in os.listdir(graph_path)]
            adjs = []
            eigvals = []
            eigvecs = []
            n_nodes = []
            max_eigval = 0
            min_eigval = 0
            for fileread in tqdm(files):
                print(f'fileread : {fileread}')
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx+1:]
                fread = os.path.join(graph_path,fileread) # edgelist ici
                fstats = os.path.join(desc_path,filen+".txt") # description correspondante là 
                print(f'fstats : {fstats}')


                """
                ici on va travailler sur l'edgelist et recueuillir les infos à obtenir 
                
                puis on travaille sur la description 

                
                """

                #load dataset to networkx
                if extension=="graphml":
                    G = nx.read_graphml(fread)
                    # Convert node labels back to tuples since GraphML stores them as strings
                    G = nx.convert_node_labels_to_integers(
                        G, ordering="sorted"
                    )
                else:
                    G = nx.read_edgelist(fread)
                # use canonical order (BFS) to create adjacency matrix
                ### BFS & DFS from largest-degree node

                
                CGs = [G.subgraph(c) for c in nx.connected_components(G)]

                # rank connected componets from large to small size
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(
                    node_degree_list, key=lambda tt: tt[1], reverse=True)

                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

                adj = torch.from_numpy(adj_bfs).float()
                diags = np.sum(adj_bfs, axis=0)
                diags = np.squeeze(np.asarray(diags))
                D = sparse.diags(diags).toarray()
                L = D - adj_bfs
                with np.errstate(divide="ignore"): # sp.errstate(divide="ignore")
                    diags_sqrt = 1.0 / np.sqrt(diags)
                diags_sqrt[np.isinf(diags_sqrt)] = 0
                DH = sparse.diags(diags).toarray()
                L = np.linalg.multi_dot((DH, L, DH))
                L = torch.from_numpy(L).float()
                eigval, eigvecs = torch.linalg.eigh(L)
                eigval = torch.real(eigval)
                eigvecs = torch.real(eigvecs)
                idx = torch.argsort(eigval)
                eigvecs = eigvecs[:,idx]

                edge_index = torch.nonzero(adj).t()

                size_diff = n_max_nodes - G.number_of_nodes()
                x = torch.zeros(G.number_of_nodes(), spectral_emb_dim+1)
                x[:,0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:,0]/(n_max_nodes-1)
                mn = min(G.number_of_nodes(),spectral_emb_dim)
                mn+=1
                x[:,1:mn] = eigvecs[:,:spectral_emb_dim]
                adj = F.pad(adj, [0, size_diff, 0, size_diff])
                adj = adj.unsqueeze(0)

                feats_stats = extract_feats(fstats)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)


                """ici data pour le training est dans stats du module du data_lst """
                data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, filename = filen))
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')
    return data_lst



#### Tackle Problem of the edge_index  updated 12.01

# import os
# import networkx as nx
# import numpy as np
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from torch_geometric.data import Data
# from scipy.sparse import diags

# from extract_feats import extract_feats, extract_numbers  # Assuming these are correctly implemented

# def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim):
#     data_lst = []
#     if dataset == 'test':
#         filename = f'./data/dataset_{dataset}_processed.pt'
#         desc_file = f'./data/{dataset}/test.txt'

#         if os.path.isfile(filename):
#             data_lst = torch.load(filename)
#             print(f'Dataset {filename} loaded from file')
#         else:
#             with open(desc_file, "r") as fr:
#                 for line in fr:
#                     line = line.strip()
#                     tokens = line.split(",")
#                     graph_id = tokens[0]
#                     desc = "".join(tokens[1:])
#                     feats_stats = extract_numbers(desc)
#                     feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
#                     data_lst.append(Data(stats=feats_stats, filename=graph_id))
#             torch.save(data_lst, filename)
#             print(f'Dataset {filename} saved')

#     else:
#         filename = f'./data/dataset_{dataset}_processed.pt'
#         graph_path = f'./data/{dataset}/graph'
#         desc_path = f'./data/{dataset}/description'

#         if os.path.isfile(filename):
#             data_lst = torch.load(filename)
#             print(f'Dataset {filename} loaded from file')
#         else:
#             files = os.listdir(graph_path)
#             for fileread in tqdm(files, desc=f"Processing {dataset} dataset"):
#                 filen = os.path.splitext(fileread)[0]
#                 fread = os.path.join(graph_path, fileread)
#                 fstats = os.path.join(desc_path, f"{filen}.txt")

#                 # Load the graph
#                 G = nx.read_graphml(fread) if fileread.endswith(".graphml") else nx.read_edgelist(fread)
#                 G = nx.convert_node_labels_to_integers(G, ordering="sorted")

#                 # Handle empty or single-node graphs
#                 if G.number_of_nodes() == 0:
#                     print(f"Warning: Empty graph in {fileread}. Skipping.")
#                     continue
#                 elif G.number_of_nodes() == 1:
#                     print(f"Single-node graph detected: {fileread}. Adding self-loop.")
#                     G.add_edge(0, 0)

#                 # BFS ordering for adjacency matrix
#                 CGs = sorted([G.subgraph(c) for c in nx.connected_components(G)], key=lambda x: x.number_of_nodes(), reverse=True)
#                 node_list_bfs = []
#                 for cg in CGs:
#                     degree_sequence = sorted(cg.degree(), key=lambda x: x[1], reverse=True)
#                     bfs_tree = nx.bfs_tree(cg, source=degree_sequence[0][0])
#                     node_list_bfs.extend(bfs_tree.nodes())
#                 adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

#                 # Adjacency matrix and edge index
#                 adj = torch.from_numpy(adj_bfs).float()
#                 edge_index = torch.nonzero(adj).t()

#                 # Validate edge_index to match padded size
#                 valid_nodes = torch.arange(G.number_of_nodes())
#                 mask = (edge_index[0] < valid_nodes.size(0)) & (edge_index[1] < valid_nodes.size(0))
#                 edge_index = edge_index[:, mask]

#                 # Compute Laplacian and eigenvectors
#                 diags = np.sum(adj_bfs, axis=0)
#                 D = np.diag(diags)
#                 L = D - adj_bfs
#                 with np.errstate(divide="ignore"):
#                     diags_sqrt = 1.0 / np.sqrt(diags)
#                 diags_sqrt[np.isinf(diags_sqrt)] = 0
#                 DH = diags_sqrt[:, None] * np.eye(G.number_of_nodes())
#                 L = DH @ L @ DH
#                 eigval, eigvecs = np.linalg.eigh(L)
#                 eigvecs = torch.from_numpy(eigvecs[:, np.argsort(eigval)]).float()

#                 # Node features (x)
#                 x = torch.zeros(n_max_nodes, spectral_emb_dim + 1)
#                 x[:G.number_of_nodes(), 0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:, 0] / (n_max_nodes - 1)
#                 x[:G.number_of_nodes(), 1:spectral_emb_dim + 1] = eigvecs[:, :spectral_emb_dim]

#                 # Pad adjacency matrix
#                 size_diff = n_max_nodes - G.number_of_nodes()
#                 adj = F.pad(adj, [0, size_diff, 0, size_diff]).unsqueeze(0)

#                 # Graph statistics
#                 feats_stats = extract_feats(fstats)
#                 feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

#                 # Append to data list
#                 data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, filename=filen))

#             torch.save(data_lst, filename)
#             print(f'Dataset {filename} saved')

#     return data_lst





# # pre processing for 

# def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim):
#     data_lst = []

#     if dataset == 'test':
#         filename = f'./data/dataset_{dataset}_3_h.pt'
#         desc_file = f'./data/{dataset}/test.txt'

#         if os.path.isfile(filename):
#             data_lst = torch.load(filename)
#             print(f'Dataset {filename} loaded from file')
#         else:
#             with open(desc_file, "r") as fr:
#                 for line in fr:
#                     line = line.strip()
#                     tokens = line.split(",")
#                     graph_id = tokens[0]
#                     desc = tokens[1:]
#                     desc = "".join(desc)
#                     feats_stats = extract_numbers(desc)
#                     feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
#                     data_lst.append(Data(stats=feats_stats, filename=graph_id))

#             torch.save(data_lst, filename)
#             print(f'Dataset {filename} saved')

#     else:
#         filename = f'./data/dataset_{dataset}_3_h.pt'
#         graph_path = f'./data/{dataset}/graph'
#         desc_path = f'./data/{dataset}/description'

#         if os.path.isfile(filename):
#             data_lst = torch.load(filename)
#             print(f'Dataset {filename} loaded from file')
#         else:
#             files = os.listdir(graph_path)
#             for fileread in tqdm(files):
#                 filen = os.path.splitext(fileread)[0]
#                 fread = os.path.join(graph_path, fileread)
#                 fstats = os.path.join(desc_path, f"{filen}.txt")

#                 # Load graph
#                 G = nx.read_edgelist(fread) if fileread.endswith(".edgelist") else nx.read_graphml(fread)
#                 G = nx.convert_node_labels_to_integers(G, ordering="sorted")

#                 # Get adjacency matrix with BFS ordering
#                 CGs = sorted([G.subgraph(c) for c in nx.connected_components(G)], key=lambda x: x.number_of_nodes(), reverse=True)
#                 node_list_bfs = []
#                 for cg in CGs:
#                     degree_sequence = sorted([(n, d) for n, d in cg.degree()], key=lambda tt: tt[1], reverse=True)
#                     bfs_tree = nx.bfs_tree(cg, source=degree_sequence[0][0])
#                     node_list_bfs.extend(bfs_tree.nodes())

#                 adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)
#                 adj = torch.from_numpy(adj_bfs).float()

#                 # Compute Laplacian and eigenvectors
#                 diags = np.sum(adj_bfs, axis=0)
#                 D = sparse.diags(diags).toarray()
#                 L = D - adj_bfs
#                 diags_sqrt = 1.0 / np.sqrt(diags, where=diags != 0)
#                 diags_sqrt[np.isinf(diags_sqrt)] = 0
#                 DH = sparse.diags(diags_sqrt).toarray()
#                 L = np.dot(np.dot(DH, L), DH)
#                 L = torch.from_numpy(L).float()

#                 eigval, eigvecs = torch.linalg.eigh(L)
#                 eigvecs = eigvecs[:, torch.argsort(eigval)]

#                 edge_index = torch.nonzero(adj).t()

#                 # Adjust features and adjacency matrix size
#                 # size_diff = n_max_nodes - G.number_of_nodes()
#                 # x = torch.zeros(G.number_of_nodes(), spectral_emb_dim + 1)
#                 # x[:, 0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:, 0] / (n_max_nodes - 1)
#                 # x[:, 1:spectral_emb_dim + 1] = eigvecs[:, :spectral_emb_dim]
#                 # adj = F.pad(adj, [0, size_diff, 0, size_diff])
#                 # adj = adj.unsqueeze(0)

#                 # # Adjust edge_index for padded nodes
#                 # edge_index = torch.nonzero(adj[:G.number_of_nodes(), :G.number_of_nodes()]).t()

#                 # # Extract graph statistics
#                 # feats_stats = extract_feats(fstats)
#                 # feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

#                 # # Validate edge_index
#                 # mask = (edge_index < G.number_of_nodes()).all(dim=0)
#                 # edge_index = edge_index[:, mask]

#                 # Adjust features and adjacency matrix size
#                 size_diff = n_max_nodes - G.number_of_nodes()
#                 x = torch.zeros(G.number_of_nodes(), spectral_emb_dim + 1)
#                 x[:, 0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:, 0] / (n_max_nodes - 1)
#                 x[:, 1:spectral_emb_dim + 1] = eigvecs[:, :spectral_emb_dim]
#                 adj = F.pad(adj, [0, size_diff, 0, size_diff])
#                 adj = adj.unsqueeze(0)

#                 # Validate and adjust edge_index
#                 valid_nodes = torch.arange(G.number_of_nodes())
#                 mask = (edge_index[0] < valid_nodes.size(0)) & (edge_index[1] < valid_nodes.size(0))
#                 edge_index = edge_index[:, mask]

#                 # Debug information
#                 print(f"Adjacency size: {adj.size()}, x size: {x.size()}, edge_index size: {edge_index.size()}")
#                 if edge_index.size(0) != 2:
#                     raise ValueError(f"Invalid edge_index shape: {edge_index.shape}. Expected [2, num_edges].")


#                 # Append data
#                 data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, filename=filen))

#             torch.save(data_lst, filename)
#             print(f'Dataset {filename} saved')

#     return data_lst




def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G



def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x




def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start





