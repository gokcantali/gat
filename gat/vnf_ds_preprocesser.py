import multiprocessing as mp

import pandas as pd

from gat.encoder import (
    boolean_string_to_int,
    ip_encoder,
    number_normalizer,
    string_encoder,
)
from gat.load_data import load_data


def enumerate_attack_classes(x):
    if x.lower() == 'benign':
        return 0
    elif 'dns spoof' in x.lower():
        return 1
    elif 'dns amp' in x.lower():
        return 2
    elif 'scan' in x.lower():
        return 3
    else:
        return 4


def diversity_index(series):
    return series.nunique() / len(series) if len(series) > 0 else 0

def process_group(group, time_window="10s"):
    group = group.set_index("timestamp")
    result = group["destination_port_label"].rolling(window=time_window).apply(diversity_index)
    group["diversity_index"] = result.values
    group.reset_index(inplace=True)
    return group

def process_chunk(chunk, time_window):
    chunk_groups = chunk.groupby("source_ip")
    processed_groups = []
    for _, group in chunk_groups:
        processed_groups.append(process_group(group, time_window))
    return pd.concat(processed_groups)

def split_dataframe(df, num_chunks):
    chunk_size = len(df) // num_chunks
    chunks = [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    return chunks

def construct_port_scan_label(df, use_diversity_index=True):
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values(by=["timestamp"], inplace=True)
    if use_diversity_index:
        num_chunks = mp.cpu_count()
        chunks = split_dataframe(df, num_chunks)
        time_window = "10s"
        with mp.Pool(processes=num_chunks) as pool:
            results = pool.starmap(process_chunk, [(chunk, time_window) for chunk in chunks])
        df = pd.concat(results, ignore_index=True)
        df["diversity_index"] = df["diversity_index"].fillna(0)
        df = df.sample(frac=1).reset_index(drop=True)
    return df

def preprocess_df(df, use_diversity_index=True):
    df = df.fillna(0)
    df["timestamp"] = (0.5 * df["start_time"] + 0.5 * df["stop_time"]).astype(int)
    df = construct_port_scan_label(df, use_diversity_index=use_diversity_index)
    return preprocess_X(df), preprocess_y(df)

def preprocess_X(df, use_diversity_index=True):
    X = df.drop(columns=["label"])
    #X["timestamp"] = pd.to_datetime(X["timestamp"], utc=True)
    #X.sort_values(by=["ip_source", "timestamp"], inplace=True)

    encoder_map = {
        "source_ip": ip_encoder,
        "destination_ip": ip_encoder,
        "source_country": string_encoder,
        "destination_country": string_encoder,
        "source_port_label": number_normalizer,
        "destination_port_label": number_normalizer,
        "source_mac": string_encoder,
        "destination_asn": string_encoder,
        "protocols": string_encoder,
        "ip_protocol": string_encoder,
        "version": string_encoder,
        "uri": string_encoder,
        "host": string_encoder,
        "hostname": string_encoder,
        "alt_name": string_encoder,
        "geo": string_encoder,
    }

    for column, encoder_function in encoder_map.items():
        X = encoder_function(X, column)
    for column in encoder_map.keys():
        X = X.drop(columns=[column], errors='ignore')
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    #get_data_insights(X, "./results/data_insights.txt")
    #visualize_data(X, "./results")

    return X


def preprocess_y(df):
    df["label"] = df["label"].apply(enumerate_attack_classes)
    return df["label"]
