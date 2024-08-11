from pathlib import Path

import pandas as pd
from torch import save, load
from torch_geometric.data import Data


def load_data(file_path: Path = Path("data/traces-3ddos-2zap-1scan.csv"), sampling_ratio=1.0):
    print("Start loading data...")
    df = pd.read_csv(file_path)
    df = df.sample(frac=sampling_ratio)
    print("Data loaded successfully.")
    return df


def save_graph_data(data: Data, name = 'traces-3ddos-2zap-1scan.pt'):
    file_path = Path(f"data/graph/{name}")
    save(data, file_path)


def load_graph_data(name = 'traces-3ddos-2zap-1scan.pt'):
    file_path = Path(f"data/graph/{name}")
    return load(file_path)
