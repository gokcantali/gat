from pathlib import Path

import pandas as pd


def load_data(file_path: Path = Path('data/traces.csv')):
    data = pd.read_csv(file_path, header=None, low_memory=False)
    header = data.iloc[0]
    data = data.drop(0)
    data.columns = header
    return data.drop(columns=['is_anomaly']), data['is_anomaly'], header
