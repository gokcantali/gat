import multiprocessing as mp
from collections import defaultdict

import pandas as pd

from gat.encoder import (
    boolean_string_to_int,
    ip_encoder,
    number_normalizer,
    string_encoder,
    hex_encoder, int_to_log1p, one_hot_encoder
)
from gat.load_data import load_data


attack_types = defaultdict(lambda: 0)


def enumerate_attack_classes(x):
    attack_types[x.lower()] += 1
    if x.lower() == 'benign':
        return 0
    elif 'malware' in x.lower():
        return 1
    elif 'scan' in x.lower():
        return 2
    elif 'udp flood' in x.lower():
        return 3
    elif 'dns amp' in x.lower():
        return 4
    elif 'dns exf' in x.lower():
        return 5
    elif 'dns spoof' in x.lower():
        return 6
    else:
        return 0


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

def preprocess_df(df, use_diversity_index=True, benign_sampling_ratio=1.0):
    df = df[(df.start_time.notna()) & (df.start_time != '')]
    df = df[(df.stop_time.notna()) & (df.stop_time != '')]
    df = df[(df.source_ip.notna()) & (df.source_ip != '')]
    df = df[(df.destination_ip.notna()) & (df.destination_ip != '')]
    df = df.fillna('')
    df["timestamp"] = (0.5 * df["start_time"] + 0.5 * df["stop_time"]).astype(int)

    # sample benign records with given ratio while preserving original order
    try:
        ratio = float(benign_sampling_ratio)
    except Exception:
        ratio = 1.0

    if 0.0 <= ratio < 1.0 and "label" in df.columns:
        benign_mask = df["label"].fillna("").str.lower() == "benign"
        benign_df = df[benign_mask]
        non_benign_df = df[~benign_mask]
        if not benign_df.empty:
            sampled_benign = benign_df.sample(frac=ratio, random_state=42).sort_index()
            df = pd.concat([non_benign_df, sampled_benign]).sort_index()

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
        "payload_source_utf8": hex_encoder,
        "payload_destination_utf8": hex_encoder,
        "source_mac": string_encoder,
        "destination_asn": string_encoder,
        "protocols": one_hot_encoder,
        "ip_protocol": one_hot_encoder,
        "version": one_hot_encoder,
        "icmp_type": one_hot_encoder,
        "uri": string_encoder,
        "host": string_encoder,
        "hostname": string_encoder,
        "alt_name": string_encoder,
        "geo": string_encoder,
        "packets": int_to_log1p,
        'data_bytes': int_to_log1p,
        'source_data_bytes': int_to_log1p,
        'destination_data_bytes': int_to_log1p,
        'bytes': int_to_log1p,
        'src_bytes': int_to_log1p,
        'dst_bytes': int_to_log1p,
        'session_length': int_to_log1p,
        'tcp_flag_syn': int_to_log1p,
        'tcp_flag_syn_ack': int_to_log1p,
        'tcp_flag_ack': int_to_log1p,
        'tcp_flag_psh': int_to_log1p,
        'tcp_flag_fin': int_to_log1p,
        'tcp_flag_rst': int_to_log1p,
        'session_segments': int_to_log1p,
        'initial_rtt': int_to_log1p
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
