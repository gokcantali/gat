from pathlib import Path

import pandas as pd


def load_data(file_path: Path = Path("data/traces.csv")):
    print("Start loading data...")
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return df
