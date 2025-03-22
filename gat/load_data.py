from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import save, load
from torch_geometric.data import Data

from .constants import TEST_SIZE

PROJECT_ROOT = Path(__file__).parent.parent


def load_data(file_path: Path = Path("data/sampled-traces-3ddos-2zap-1scan.csv"), sampling_ratio=1.0):
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


def create_subset_from_dataset_using_monte_carlo(file_name: str):
    print("File Name: ", file_name)
    file_path = Path(f"{PROJECT_ROOT}/data/{file_name}")

    # Load the data
    df = load_data(file_path=file_path, sampling_ratio=1.0)

    # Randomly sample a subset of the data using random ratios for each class
    benign_ratio = np.random.uniform(0.2, 1)
    dos_ratio = np.random.uniform(0.3, 1)
    port_scan_ratio = np.random.uniform(0.5, 1)
    zap_scan_ratio = np.random.uniform(1, 1)

    # Sample the data based on the drawn ratios
    df_benign = df[df["anomaly_class"] == 0].sample(frac=benign_ratio)
    df_dos = df[df["anomaly_class"] == 1].sample(frac=dos_ratio)
    df_port_scan = df[df["anomaly_class"] == 2].sample(frac=port_scan_ratio)
    df_zap_scan = df[df["anomaly_class"] == 3].sample(frac=zap_scan_ratio)

    # Combine the dataframes
    df_subset = pd.concat([df_benign, df_dos, df_port_scan, df_zap_scan])

    # Sort by timestamp
    df_subset = df_subset.sort_values(by=['timestamp'])

    # Construct the file name based on the ratios
    file_name = "traces-"
    file_name += f"benign{benign_ratio:.2f}-"
    file_name += f"dos{dos_ratio:.2f}-"
    file_name += f"port{port_scan_ratio:.2f}-"
    file_name += f"zap{zap_scan_ratio:.2f}"

    # Save the training and testing subsets to a new file
    train_file_name = file_name + "-train.csv"
    test_file_name = file_name + "-test.csv"

    df_subset_train, df_subset_test = train_test_split(df_subset, test_size=TEST_SIZE)
    df_subset_train.to_csv(f"{PROJECT_ROOT}/data/subsample/{train_file_name}", index=False)
    df_subset_test.to_csv(f"{PROJECT_ROOT}/data/subsample/{test_file_name}", index=False)

    return df_subset


# create_subset_from_dataset_using_monte_carlo(Path(f"{PROJECT_ROOT}/data/sampled-traces-3ddos-2zap-1scan.csv"))
