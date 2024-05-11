from pathlib import Path

import pandas as pd


def load_data(file_path: Path = Path('data/traces-3.csv')):
    return pd.read_csv(file_path)
