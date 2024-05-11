import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

def construct_port_scan_label(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.sort_values(by=['ip_source', 'timestamp'], inplace=True)
    time_window = '10S'  

    def diversity_index(df):
        if not isinstance(df, pd.Series):
            df = pd.Series(df)
        return df.nunique() / len(df) if len(df) > 0 else 0

    df.set_index('timestamp', inplace=True)
    results = df.groupby('ip_source')['destination_port_label'].rolling(
        window=time_window).apply(diversity_index, raw=False)
    df['diversity_index'] = results.values
    df.reset_index(inplace=True)
    df.drop(columns=['timestamp'], inplace=True)
    df['diversity_index'] = df['diversity_index'].fillna(0)
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    return df


from sklearn.neighbors import kneighbors_graph

def create_knn_graph(X, k=5):
    A = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=True)
    A = A.tocoo()  # Convert to COO format
    row = A.row.astype(np.int64)
    col = A.col.astype(np.int64)
    edge_index = np.vstack([row, col])
    return torch.tensor(edge_index, dtype=torch.long)


def convert_to_graph(X, y):
    #X = X[['diversity_index']]
    edge_index = create_knn_graph(X)
    data = Data(x=torch.tensor(X.values, dtype=torch.float), 
                      edge_index=edge_index, 
                      y=torch.tensor(y.values, dtype=torch.long))

    return data
