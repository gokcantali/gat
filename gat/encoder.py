import hashlib

import pandas as pd
import torch


def ip_encoder(df, column_name):
    def encode_ip(ip_address):
        # Check if the address is IPv4 or IPv6
        if ':' in ip_address:  # This suggests an IPv6 address
            # Split by colon and convert hex to int
            parts = [int(x, 16) if x else 0 for x in ip_address.split(':')]
            # IPv6 might have less than 8 parts due to '::' compression, pad with zeros
            parts += [0] * (8 - len(parts))
        elif '.' in ip_address:  # This suggests an IPv4 address
            # Split by dot and convert to int
            parts = [int(x) for x in ip_address.split('.')]
        else:
            raise ValueError("Invalid IP address format")
        return parts

    # Apply the encode_ip function to the specified column and create a new DataFrame
    ip_encoded = df[column_name].apply(encode_ip)

    # Convert list of parts into separate columns
    ip_df = pd.DataFrame(ip_encoded.tolist(), columns=[f"{column_name}_part{i}" for i in range(1, 9)])

    # Drop the original IP address column and concatenate the new DataFrame
    df = df.drop(column_name, axis=1).join(ip_df)

    return df


def string_encoder(df, column_name):
    # Convert string to a consistent hash value
    def get_hash(x):
        if pd.notna(x):
            return int(hashlib.sha256(x.encode()).hexdigest(), 16)
        else:
            return 0  # Handle NaN values

    # Apply the hashing function and convert to a PyTorch tensor
    hashed_values = df[column_name].apply(get_hash)
    tensor = torch.tensor(hashed_values.values.astype(float))

    # Normalize the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    # Store normalized values back in DataFrame
    df[f'{column_name}_normalized'] = normalized_tensor.numpy()

    # Drop the original and hashed columns
    df = df.drop(column_name, axis=1)
    df = df.drop(f'{column_name}_hashed', axis=1, errors='ignore')  # Use errors='ignore' to handle cases where column might not exist

    return df

def number_normalizer(df, column_name):
    # Convert string representations of numbers to integers
    df[f'{column_name}_int'] = df[column_name].apply(lambda x: int(x) if pd.notna(x) else None)

    # Convert to a PyTorch tensor
    tensor = torch.tensor(df[f'{column_name}_int'].dropna().values.astype(float))

    # Normalize the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    # Assign the normalized values back to the DataFrame
    # We need to handle NaN entries carefully, re-aligning with the original DataFrame index
    normalized_series = pd.Series(normalized_tensor.numpy(), index=df[df[column_name].notna()].index)
    df[f'{column_name}_normalized'] = normalized_series

    # Drop the original and intermediate integer columns
    df = df.drop([column_name, f'{column_name}_int'], axis=1)

    return df

def boolean_string_to_int(df, column = None):
    # Convert "True"/"False" strings to integers
    if column:
        mapping = {'False': 0, 'True': 1}
        df[column] = df[column].map(mapping).astype(int)
    else:
        mapping = {'False': 0, 'True': 1}
        df = df.map(mapping).astype(int)
    return df
