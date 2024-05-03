import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from gat.encoder import (
    boolean_string_to_int,
    ip_encoder,
    number_normalizer,
    string_encoder,
)


def construct_port_scan_label(X):
    X['timestamp'] = pd.to_datetime(X['timestamp'], utc=True)
    X.sort_values(by=['ip_source', 'timestamp'], inplace=True)
    time_window = '5T'  

    def diversity_index(x):
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        return x.nunique() / len(x) if len(x) > 0 else 0

    X.set_index('timestamp', inplace=True)
    results = X.groupby('ip_source')['destination_port_label'].rolling(
        window=time_window).apply(diversity_index, raw=False)
    X['diversity_index'] = results.values
    X.reset_index(inplace=True)
    X.drop(columns=['timestamp'], inplace=True)
    X['diversity_index'] = X['diversity_index'].fillna(0)
    # shuffle
    X = X.sample(frac=1).reset_index(drop=True)
    return X


def convert_to_graph(X, y):
    X = construct_port_scan_label(X)
    encoder_map = {
        'ip_source': ip_encoder,
        'ip_destination': ip_encoder,
        'source_pod_label': string_encoder,
        'destination_pod_label': string_encoder,
        'source_namespace_label': string_encoder,
        'destination_namespace_label': string_encoder,
        'source_port_label': number_normalizer,
        'destination_port_label': number_normalizer,
        'ack_flag': boolean_string_to_int,
        'psh_flag': boolean_string_to_int
    }
    for column, encoder_function in encoder_map.items():
        X = encoder_function(X, column)
    
    # Create a DataFrame with the correct column names
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Convert the DataFrame to a tensor
    X_tensor = torch.tensor(X.values, dtype=torch.float)

    # Convert the label data to a tensor
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    # Total number of nodes calculated from the number of rows in the tensor
    num_nodes = len(X_tensor)

    # Initialize masks as boolean tensors for training, testing, and validation
    masks = [torch.zeros(num_nodes, dtype=torch.bool) for _ in range(3)]
    limits = [int(0.8 * num_nodes), int(0.9 * num_nodes), num_nodes]

    # Assign True up to the specified limit for each mask and shuffle
    for mask, limit in zip(masks, limits):
        mask[:limit] = True
        np.random.shuffle(mask.numpy())

    # Defines a simple bi-directional connection between two consecutive nodes
    edge_index = torch.tensor([[i, i+1] for i in range(num_nodes-1) for _ in (0, 1)], dtype=torch.long).t().contiguous()

    # Return a Data object containing node features, edge connections, labels, and masks
    return Data(x=X_tensor, edge_index=edge_index, y=y_tensor,
                train_mask=masks[0], test_mask=masks[1], val_mask=masks[2])
