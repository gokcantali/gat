import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data


def create_knn_graph(X, k=5):
    A = kneighbors_graph(
        X.values, n_neighbors=k, mode="connectivity",
        include_self=True, n_jobs=-1)
    A = A.tocoo()
    row = A.row.astype(np.int64)
    col = A.col.astype(np.int64)
    edge_index = np.vstack([row, col])
    return torch.tensor(edge_index, dtype=torch.long)


def convert_to_graph(X, y):
    edge_index = create_knn_graph(X)
    data = Data(x=torch.tensor(X.values, dtype=torch.float),
                      edge_index=edge_index,
                      y=torch.tensor(y.values, dtype=torch.long))

    return data
