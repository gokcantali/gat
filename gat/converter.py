import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy import average
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, RandomNodeLoader
from torch_geometric.transforms.line_graph import LineGraph

from gat.encoder import ip_encoder, string_encoder
from gat.load_data import load_data, save_graph_data
from gat.preprocesser import preprocess_X, construct_port_scan_label

PROJECT_ROOT = Path(__file__).parent.parent


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
        if source not in node_to_source_edge_labels:
            node_to_source_edge_labels[source] = []
        node_to_source_edge_labels[source].append(label)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    node_labels = [0] * len(nodes)
    for ind, node in enumerate(nodes):
        if average(node_to_source_edge_labels.get(node, [0])) > 0.75:
            node_labels[ind] = 1

    node_df = ip_encoder(node_df, 'ip', False)
    node_df['port'] = node_df['port'].replace('', 0).fillna(0).astype(int)

    edge_attr = X_window[[
        "ip_source", "port_source",
        "ip_destination", "port_destination",
        "ack_flag", "psh_flag",
        "source_pod_label", "destination_pod_label",
        "source_namespace_label", "destination_namespace_label",
    ]]
    edge_attr = ip_encoder(edge_attr, "ip_source", False)
    edge_attr = ip_encoder(edge_attr, "ip_destination", False)
    edge_attr['port_source'] = edge_attr['port_source'].replace('', 0).fillna(0).astype(int)
    edge_attr['port_destination'] = edge_attr['port_destination'].replace('', 0).fillna(0).astype(int)
    edge_attr['ack_flag'] = edge_attr['ack_flag'].replace({"True": 1, "False": 0}).fillna(0).astype(int)
    edge_attr['psh_flag'] = edge_attr['psh_flag'].replace({"True": 1, "False": 0}).fillna(0).astype(int)
    edge_attr = string_encoder(edge_attr, 'source_pod_label')
    edge_attr = string_encoder(edge_attr, 'destination_pod_label')
    edge_attr = string_encoder(edge_attr, 'source_namespace_label')
    edge_attr = string_encoder(edge_attr, 'destination_namespace_label')

    lg = LineGraph(force_directed=True)
    inverted_graph = lg(Data(
        x=torch.tensor(node_df.values, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr.values, dtype=torch.float),
        y=torch.tensor(edge_labels, dtype=torch.long)
    ))
    inverted_graph.num_nodes = inverted_graph.x.shape[0]
    return inverted_graph


def create_tdg_graphs_using_window(df, window_size='5Min'):
    df_list = [g for _, g in df.groupby(pd.Grouper(key='timestamp', freq=window_size))]

    return DataLoader(
        [
            # create_tdg_graph(
            #     X_window=df_window.drop(columns=['is_anomaly']),
            #     y_window=df_window['is_anomaly'].replace({"True": 1, "False": 0}).astype(int)
            # ) for df_window in df_list if df_window.shape[0] > 0
            convert_to_graph(
                X=preprocess_X(df_window),
                y=df_window['is_anomaly'].replace({"True": 1, "False": 0}).astype(int)
            ) for df_window in df_list if df_window.shape[0] > 0
        ],
    )


def create_randomly_partitioned_tdg_graphs(df):
    return RandomNodeLoader(
        create_tdg_graph(
            X_window=df.drop(columns=['is_anomaly']),
            y_window=df['is_anomaly'].replace({"True": 1, "False": 0}).astype(int)
        ),
        num_parts=300,
        shuffle=True
    )


def create_randomly_partitioned_knn_graphs(df):
    graph = convert_to_graph(
        X=preprocess_X(df),
        y=df['is_anomaly'].replace({"True": 1, "False": 0}).astype(int)
    )

    return RandomNodeLoader(graph, num_parts=150, shuffle=True)


def create_graph_from_dataset(dataset_file_path: Path):
    df = load_data(file_path=dataset_file_path, sampling_ratio=1)
    df = construct_port_scan_label(df, use_diversity_index=True)
    df["is_anomaly"] = df["is_anomaly"].replace({"True": 1, "False": 0}).astype(int)
    graph = convert_to_graph(
        X=preprocess_X(df, use_diversity_index=True),
        y=df['is_anomaly'].replace({"True": 1, "False": 0}).astype(int)
    )
    return graph


def create_one_graph_from_the_first_existing_dataset_subsample():
    dataset_subsample_list = os.listdir(f"{PROJECT_ROOT}/data/subsample")
    dataset_graph_list = os.listdir(f"{PROJECT_ROOT}/data/subsample/graph")

    for ds_file_name in dataset_subsample_list:
        if not ds_file_name.endswith(".csv"):
            continue

        graph_file_name = ds_file_name.replace(".csv", ".pt")
        if graph_file_name in dataset_graph_list:
            continue

        graph = create_graph_from_dataset(
            dataset_file_path=Path(f"{PROJECT_ROOT}/data/subsample/{ds_file_name}"),
        )
        save_graph_data(graph, f"{PROJECT_ROOT}/data/subsample/graph/{graph_file_name}")
