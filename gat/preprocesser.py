import pandas as pd

from gat.converter import construct_port_scan_label
from gat.encoder import ip_encoder, string_encoder, number_normalizer, boolean_string_to_int
from gat.load_data import load_data


def get_data_insights(df):
    print("Data info:")
    print("----------------------------------------------")
    print(df.info())
    print("Data describe:")
    print("----------------------------------------------")
    print(df.describe())
    print("Data nunique:")
    print("----------------------------------------------")
    print(df.nunique())
    print("Data correlation:")
    print("----------------------------------------------")
    print(df.corr())
    print("Data skew:")
    print("----------------------------------------------")
    print(df.skew())
    print("Data kurt:")
    print("----------------------------------------------")
    print(df.kurt())


def preprocess_df():
    df = load_data()
    df = construct_port_scan_label(df)
    df['is_anomaly'] = df['is_anomaly'].replace({'True': 1, 'False': 0}).astype(int)
    return df

def preprocess_X(df):
    X = df.drop(columns=['is_anomaly'])

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
        
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    get_data_insights(X)


    features = ['ack_flag', 'psh_flag', 'diversity_index', 'ip_source_part1',
                'ip_source_part2', 'ip_source_part3', 'ip_source_part4',
                'ip_source_part5', 'ip_source_part6', 'ip_source_part7',
                'ip_source_part8', 'ip_destination_part1', 'ip_destination_part2',
                'ip_destination_part3', 'ip_destination_part4', 'ip_destination_part5',
                'ip_destination_part6', 'ip_destination_part7', 'ip_destination_part8',
                'source_pod_label_normalized', 'destination_pod_label_normalized',
                'source_namespace_label_normalized',
                'destination_namespace_label_normalized',
                'source_port_label_normalized', 'destination_port_label_normalized']
    
    # bad:
    # ack ps
    # pod label
    # namespace label
    


    features_2 = ['ack_flag', 'psh_flag', 'diversity_index', 'ip_source_part1',
                'ip_source_part2', 'ip_source_part3', 'ip_source_part4',
                'ip_source_part5', 'ip_source_part6', 'ip_source_part7',
                'ip_source_part8', 'ip_destination_part1', 'ip_destination_part2',
                'ip_destination_part3', 'ip_destination_part4', 'ip_destination_part5',
                'ip_destination_part6', 'ip_destination_part7', 'ip_destination_part8',
                'source_pod_label_normalized', 'destination_pod_label_normalized',
                'source_namespace_label_normalized',
                'destination_namespace_label_normalized',
                'source_port_label_normalized', 'destination_port_label_normalized']
    
    X = X[features]
    
    
    
    
    return X
        
def preprocess_y(df):
    return df['is_anomaly']
