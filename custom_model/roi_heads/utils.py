import torch
import torch.nn.functional as F
from torch_geometric.data import Data

def cosine_similarity_matrix(tensor):
    # tensor de taille [8, 2048, 32, 32]
    batch_size, num_channels, height, width = tensor.shape
    # Reshape le tenseur pour avoir des vecteurs de caractéristiques par pixel
    tensor = tensor.permute(0, 2, 3, 1).reshape(batch_size, height * width, num_channels)
    adjacency_matrices = []
    for i in range(batch_size):
        features = tensor[i]  # features de taille [1024, 2048]
        # Normalisation L2 des vecteurs de caractéristiques
        features_norm = F.normalize(features, p=2, dim=1)
        # Calcul de la matrice de similarité cosinus
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        adjacency_matrices.append(similarity_matrix)
    return adjacency_matrices


def normalize_adjacency_matrix(adjacency_matrix):
    # Ajout d'une matrice identité pour les self-loops
    adjacency_matrix = adjacency_matrix + torch.eye(adjacency_matrix.size(0)).to(adjacency_matrix.device)
    # Calculer le degré de chaque nœud
    degree = torch.sum(adjacency_matrix, axis=1)
    # Ajouter une petite valeur epsilon pour éviter la division par zéro
    epsilon = 1e-6
    degree = degree + epsilon
    # Calculer la racine carrée inverse du degré
    degree_inv_sqrt = torch.pow(degree, -0.5)
    # Créer une matrice diagonale pour le degré inverse sqrt
    degree_inv_sqrt_matrix = torch.diag(degree_inv_sqrt)
    # Normaliser la matrice d'adjacence
    normalized_adjacency_matrix = torch.mm(torch.mm(degree_inv_sqrt_matrix, adjacency_matrix), degree_inv_sqrt_matrix)
    return normalized_adjacency_matrix


def create_graph_data(features, adj):
    # Création de l'objet Data de PyTorch Geometric
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    edge_weight = adj[edge_index[0], edge_index[1]]
    data = Data(x=features, edge_index=edge_index, edge_attr=edge_weight)
    return data