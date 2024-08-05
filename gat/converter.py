import numpy as np
import pandas as pd
import torch
from numpy import average
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gat.encoder import ip_encoder


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

def create_tdg_graph(X_window, y_window):
    X_window['port_source'] = X_window['source_port_label'].fillna(0).astype(int).astype(str)
    X_window['port_destination'] = X_window['destination_port_label'].fillna(0).astype(int).astype(str)

    nodes = pd.concat([
        pd.Series(X_window['ip_source'] + ':' + X_window['port_source']),
        pd.Series(X_window['ip_destination'] + ':' + X_window['port_destination'])
    ], axis=0).unique()
    node_df = pd.DataFrame([(node.split(":")[0], node.split(":")[1]) for node in nodes], columns=['ip', 'port'])
    node_mapping = {node: idx for idx, node in enumerate(nodes)}

    edges = []
    edge_labels = []
    node_to_source_edge_labels = {}
    for index, row in X_window.iterrows():
        label = y_window[index]
        edge_labels.append(label)

        source = f"{row['ip_source']}:{row['port_source']}"
        destination = f"{row['ip_destination']}:{row['port_destination']}"
        edges.append([node_mapping[source], node_mapping[destination]])
        edge_labels.append(label)
        if source not in node_to_source_edge_labels:
            node_to_source_edge_labels[source] = []
        node_to_source_edge_labels[source].append(label)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    node_labels = [0] * len(nodes)
    for ind, node in enumerate(nodes):
        if average(node_to_source_edge_labels.get(node, [0])) > 0.5:
            node_labels[ind] = 1

    node_df = ip_encoder(node_df, 'ip', False)
    node_df['port'] = node_df['port'].replace('', 0).fillna(0).astype(int)

    return Data(
        x=torch.tensor(node_df.values, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_labels, dtype=torch.float),
        y=torch.tensor(node_labels)
    )


def create_tdg_graphs_using_window(df, window_size='1Min'):
    df_list = [g for _, g in df.groupby(pd.Grouper(key='timestamp', freq=window_size))]
    return DataLoader(
        [
            create_tdg_graph(
                X_window=df_window.drop(columns=['is_anomaly']),
                y_window=df_window['is_anomaly'].replace({"True": 1, "False": 0}).astype(int)
            ) for df_window in df_list
        ],
    )
