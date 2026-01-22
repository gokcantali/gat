import time
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import torch
from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from gat.load_data import load_data
from gat.vnf_ds_preprocesser import preprocess_df, preprocess_X, preprocess_y
from gat.rnn import NetworkTrafficRNN, pad_collate_fn, SequenceDataset, subset


RANDOM_STATE = 56
TEST_RATIO = 0.10
VALIDATION_RATIO = 0.10
TRIALS = 10


class RandomForestConfig:
    max_depth = 4
    number_of_estimators = 100

    def __init__(self, **kwargs):
        if 'max_depth' in kwargs:
            self.max_depth = kwargs['max_depth']
        if 'n_estimators' in kwargs:
            self.n_estimators = kwargs['n_estimators']

def report_cm_results(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "accuracy": (tp + tn) / (tp + tn + fn + fp),
        "f1": 2 * tp / (2 * tp + fp + fn),
    }


def train_random_forest(train_data, train_label, **kwargs):
    clf = RandomForestClassifier(**kwargs)
    clf.fit(train_data, train_label)
    return clf


def test_random_forest(clf, test_data, test_label):
    predicted_label = clf.predict(test_data)
    return confusion_matrix(test_label, predicted_label)


def train_svm(train_data, train_label, **kwargs):
    clf = LinearSVC(**kwargs)
    clf.fit(train_data, train_label)
    return clf


def test_svm(clf, test_data, test_label):
    predicted_label = clf.predict(test_data)
    return confusion_matrix(test_label, predicted_label)


def train_random_forest_with_k_fold_cv(X_train_val, y_train_val, is_verbose=True):
    parameters = {
        "max_depth": [3, 4, 5, 6],
        "n_estimators": [50, 100, 150],
    }
    n_splits = int((1 - TEST_RATIO) / VALIDATION_RATIO)

    start_time = time.time()
    skf = StratifiedKFold(n_splits=n_splits)

    best_score = 0.0
    best_config = {}
    for max_depth in parameters["max_depth"]:
        for n_estimators in parameters["n_estimators"]:
            rfc = RandomForestClassifier(
                max_depth=max_depth, n_estimators=n_estimators
            )

            scores = []
            for ind, (train_ind, val_ind) in enumerate(skf.split(X_train_val, y_train_val)):
                X_train = X_train_val[X_train_val.index.isin(train_ind)]
                y_train = y_train_val[y_train_val.index.isin(train_ind)]
                X_val = X_train_val[X_train_val.index.isin(val_ind)]
                y_val = y_train_val[y_train_val.index.isin(val_ind)]

                rfc.fit(X_train, y_train)

                score = rfc.score(X_val, y_val)
                scores.append(score)

            if mean(scores) > best_score:
                best_score = mean(scores)
                best_config = {
                    "max_depth": max_depth,
                    "n_estimators": n_estimators,
                }

    end_time = time.time()

    if is_verbose:
        print(f"Cross validation time: {end_time - start_time} seconds")
        print("Best Parameters based on Grid Search:")
        print(best_config)

    clf = RandomForestClassifier(**best_config)
    clf.fit(X_train_val, y_train_val)

    return clf


def train_svm_with_k_fold_cv(X_train_val, y_train_val, is_verbose=True):
    parameters = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "tol": [1e-4, 1e-3],
    }
    n_splits = int((1 - TEST_RATIO) / VALIDATION_RATIO)
    skf = StratifiedKFold(n_splits=n_splits)

    start_time = time.time()

    best_score = 0.0
    best_config = {}
    for c in parameters["C"]:
        for penalty in parameters["penalty"]:
            for tol in parameters["tol"]:
                svc = LinearSVC(
                    C=c,
                    penalty=penalty,
                    tol=tol,
                    dual=True
                )

                scores = []
                for ind, (train_ind, val_ind) in enumerate(skf.split(X_train_val, y_train_val)):
                    X_train = X_train_val[X_train_val.index.isin(train_ind)]
                    y_train = y_train_val[y_train_val.index.isin(train_ind)]
                    X_val = X_train_val[X_train_val.index.isin(val_ind)]
                    y_val = y_train_val[y_train_val.index.isin(val_ind)]

                    svc.fit(X_train, y_train)

                    score = svc.score(X_val, y_val)
                    scores.append(score)

                if mean(scores) > best_score:
                    best_score = mean(scores)
                    best_config = {
                        "C": c,
                        "penalty": penalty,
                        "tol": tol
                    }

    end_time = time.time()

    if is_verbose:
        print(f"Cross validation time: {end_time - start_time} seconds")
        print("Best Parameters based on Grid Search:")
        print(best_config)

    svc = LinearSVC(**best_config, dual=True)
    svc.fit(X_train_val, y_train_val)

    return svc


def create_hyper_param_space_for_rnn():
    """ Constructs hyperparameter configurations for RNN models """
    from itertools import product

    hyper_param_space = {
        "learning_rate": [0.001],
        "batch_size": [32, 64],
        "num_epochs": [20],
        "patience": [3],
        "num_layers": [2],
        "hidden_size": [64, 128],
        "dropout": [0.05],
    }

    # Create a list of all combinations of hyperparameters
    hyper_param_combs = []
    for values in product(*hyper_param_space.values()):
        hyper_param_combination = {}
        for i, key in enumerate(hyper_param_space.keys()):
            hyper_param_combination[key] = values[i]
        hyper_param_combs.append(hyper_param_combination)

    return hyper_param_combs


def create_time_window_sequences(data, window_seconds=0.001, stride_seconds=0.0005, label_reducer=None, max_sequences=10000):
    """
    Create sequences based on time windows from DataFrame with timestamp column.

    Args:
        data: DataFrame with features and label columns
        window_seconds: Size of window in seconds (for VNF data, try 0.001 as timestamps might be in ms)
        stride_seconds: Stride between windows in seconds
        label_reducer: Function to reduce labels in a window to a single value
                      (e.g., majority vote, most common, etc.)
        max_sequences: Maximum number of sequences to create (for performance)

    Returns:
        X: List of feature sequences
        y: List of corresponding labels
    """
    if label_reducer is None:
        # Use most common label, but prioritize anomalies (label 1) to avoid class imbalance
        def prioritize_anomaly(labels):
            if 1 in labels:
                count_ones = sum(1 for l in labels if l == 1)
                if count_ones >= len(labels) * 0.3:  # If at least 30% are anomalies
                    return 1
            return Counter(labels).most_common(1)[0][0]

        label_reducer = prioritize_anomaly

    # Ensure data has a timestamp column
    if 'timestamp' not in data.columns:
        raise ValueError("Data must contain a 'timestamp' column")

    # Sort by timestamp
    data = data.sort_values('timestamp').reset_index(drop=True)

    # Extract features and labels
    features = data.drop(columns=['label', 'timestamp'])
    labels = data['label']
    timestamps = data['timestamp'].values

    # Convert timestamps to a normalized scale
    # Determine scale based on timestamp range - if large values, might be microseconds
    min_timestamp = timestamps.min()
    max_timestamp = timestamps.max()
    timestamp_range = max_timestamp - min_timestamp

    # Auto-detect time scale
    time_scale = 1.0  # default seconds
    if timestamp_range > 10000000:  # If range is very large (millions), likely microseconds
        time_scale = 0.000001
        print("Detected microsecond timestamps - adjusting window scale")
    elif timestamp_range > 10000:  # If range is large (thousands), likely milliseconds
        time_scale = 0.001
        print("Detected millisecond timestamps - adjusting window scale")

    # Adjust window and stride to the detected timescale
    adjusted_window = window_seconds / time_scale
    adjusted_stride = stride_seconds / time_scale

    print(f"Original window: {window_seconds}s, stride: {stride_seconds}s")
    print(f"Detected time scale: {time_scale}s")
    print(f"Adjusted window: {adjusted_window}, stride: {adjusted_stride}")

    timestamps_norm = timestamps - min_timestamp

    # Create sequences using sliding windows
    X, y = [], []

    # Iterate through possible window start times
    current_time = 0
    max_time = timestamps_norm.max()
    seq_count = 0

    # Track class distribution
    class_counts = {0: 0, 1: 0}

    print(f"Time range: 0 to {max_time} units")

    while current_time < max_time and seq_count < max_sequences:
        end_time = current_time + adjusted_window

        # Find all indices within the current window
        window_indices = np.where((timestamps_norm >= current_time) &
                                 (timestamps_norm < end_time))[0]

        if len(window_indices) > 1:  # Ensure we have at least 2 points in the window
            window_features = features.iloc[window_indices].values
            window_labels = labels.iloc[window_indices].values

            # For very long sequences, truncate to improve performance
            if len(window_features) > 200:
                # Sample every nth element to reduce length while preserving pattern
                n = len(window_features) // 200 + 1
                window_features = window_features[::n]

            # Get the label for this sequence
            sequence_label = label_reducer(window_labels)
            class_counts[sequence_label] = class_counts.get(sequence_label, 0) + 1

            X.append(window_features)
            y.append(sequence_label)
            seq_count += 1

        # Move window forward by stride amount
        current_time += adjusted_stride

    print(f"Created {len(X)} sequences from {len(data)} data points")
    print(f"Class distribution in sequences: {class_counts}")
    print(f"Average sequence length: {np.mean([len(seq) for seq in X]) if X else 0:.2f} points")
    return X, y


def create_fixed_length_sequences(data, sequence_length=10, stride=5, max_sequences=5000, balance_classes=True):
    """
    Create sequences of fixed length from DataFrame, without relying on timestamps.

    Args:
        data: DataFrame with features and label columns
        sequence_length: Number of consecutive items in each sequence
        stride: Number of items to slide forward when creating the next sequence
        max_sequences: Maximum number of sequences to create (for performance)
        balance_classes: Whether to balance the class distribution in generated sequences

    Returns:
        X: List of feature sequences
        y: List of corresponding labels
    """
    # Ensure data has a 'label' column
    if 'label' not in data.columns:
        raise ValueError("Data must contain a 'label' column")

    # Extract features and labels
    features = data.drop(columns=['label'])
    if 'timestamp' in features.columns:
        features = features.drop(columns=['timestamp'])

    labels = data['label']

    # Create sequences using sliding windows of fixed length
    X_normal, y_normal = [], []  # For label 0 (normal)
    X_anomaly, y_anomaly = [], []  # For other labels (anomaly)

    # Track class distribution
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    # Create sequences
    seq_count = 0
    for i in range(0, len(data) - sequence_length + 1, stride):
        if seq_count >= max_sequences and not balance_classes:
            break

        # Extract sequence
        window_features = features.iloc[i:i+sequence_length].values
        window_labels = labels.iloc[i:i+sequence_length].values

        anomaly_ratio = int(np.count_nonzero(window_labels)) / len(window_labels)

        # Determine the anomaly ratio threshold by the proportion of anomalies in the dataset
        nonzero_count = int(np.count_nonzero(labels))
        dataset_anomaly_ratio = nonzero_count / len(labels) if len(labels) > 0 else 0.0

        # Determine sequence label
        if anomaly_ratio > dataset_anomaly_ratio:
            # find the majority label in window_labels except zero
            non_zero_labels = window_labels[window_labels != 0]
            if len(non_zero_labels) == 0:
                # no non-zero labels present — fallback to anomaly label 1
                sequence_label = 1
            else:
                sequence_label = int(Counter(non_zero_labels).most_common(1)[0][0])

            X_anomaly.append(window_features)
            y_anomaly.append(sequence_label)
        else:
            sequence_label = 0
            X_normal.append(window_features)
            y_normal.append(sequence_label)

        # Update class distribution counter
        class_counts[sequence_label] = class_counts.get(sequence_label, 0) + 1
        seq_count += 1

    # Balance classes if requested
    if balance_classes:
        # Determine the minimum number of sequences per class
        min_class_count = min(len(X_normal), len(X_anomaly))
        if min_class_count == 0:
            # If one class has no sequences, just use what we have
            X = X_normal + X_anomaly
            y = y_normal + y_anomaly
        else:
            # Randomly select min_class_count sequences from each class
            np.random.seed(RANDOM_STATE)  # For reproducibility
            normal_indices = np.random.choice(len(X_normal), min(min_class_count*3, len(X_normal)), replace=False)
            anomaly_indices = np.random.choice(len(X_anomaly), min(min_class_count*3, len(X_anomaly)), replace=False)

            X = [X_normal[i] for i in normal_indices] + [X_anomaly[i] for i in anomaly_indices]
            y = [y_normal[i] for i in normal_indices] + [y_anomaly[i] for i in anomaly_indices]

            # Shuffle the balanced dataset
            combined = list(zip(X, y))
            np.random.shuffle(combined)
            X, y = zip(*combined)
            X, y = list(X), list(y)

            # Update class counts
            class_counts = {0: len(normal_indices), 1: len(anomaly_indices)}
    else:
        # Just combine the sequences
        X = X_normal + X_anomaly
        y = y_normal + y_anomaly

    print(f"Created {len(X)} sequences from {len(data)} data points")
    print(f"Sequence length: {sequence_length}, Stride: {stride}")
    print(f"Class distribution in sequences: {class_counts}")

    return X, y


def train_rnn_model(X_train, y_train, X_val, y_val, rnn_cell="LSTM", hyperparams=None, ext_model=None):
    """Train an RNN model on the VNF data"""
    if len(X_train) == 0:
        raise ValueError("Empty training data")

    if hyperparams is None:
        hyperparams = {}

    # Get input size from the first sequence's feature dimension
    input_size = X_train[0].shape[1]

    # Count number of classes
    num_classes = 4

    # Create model
    model = NetworkTrafficRNN(
        input_size=input_size,
        hidden_size=hyperparams.get('hidden_size', 128),
        num_layers=hyperparams.get('num_layers', 2),
        num_classes=num_classes,
        cell_type=rnn_cell,
        dropout=hyperparams.get('dropout', 0.05),
        hyperparams=hyperparams
    )
    if ext_model is not None:
        model.load_state_dict(ext_model.state_dict())

    # Train the model
    loss, f1_score = model.train_rnn(X_train, y_train, X_val, y_val)

    return model, loss, f1_score


def evaluate_rnn_model(model, X_test, y_test, ext_model=None):
    """Evaluate the RNN model"""
    if ext_model is not None:
        model.load_state_dict(ext_model.state_dict())
    loss, f1, report = model.evaluate_rnn(X_test, y_test)
    return loss, f1, report


def train_rnn_with_k_fold_cv(X_sequences, y_sequences, rnn_cell="LSTM", is_verbose=True):
    """
    Train RNN model with k-fold cross-validation for hyperparameter tuning
    """
    hyper_param_combinations = create_hyper_param_space_for_rnn()
    n_splits = int((1 - TEST_RATIO) / VALIDATION_RATIO)

    start_time = time.time()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    best_score = 0.0
    best_params = None
    best_model = None

    # Convert sequences to numpy arrays if they're not already
    y_seq_array = np.array(y_sequences)

    for hyper_param_conf in hyper_param_combinations:
        if is_verbose:
            print(f"Testing hyperparameters: {hyper_param_conf}")

        # Save initial random weights to ensure fair comparison
        initial_model = NetworkTrafficRNN(
            input_size=X_sequences[0].shape[1],
            hidden_size=hyper_param_conf['hidden_size'],
            num_layers=hyper_param_conf['num_layers'],
            num_classes=4,
            cell_type=rnn_cell,
            dropout=hyper_param_conf['dropout'],
            hyperparams=hyper_param_conf
        )
        torch.save(initial_model.state_dict(), 'initial_rnn_vnf_model.pt')

        avg_f1_score = 0.0
        fold_count = 0

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_seq_array)), y_seq_array)):
            fold_count += 1
            X_fold_train = [X_sequences[i] for i in train_idx]
            y_fold_train = [y_sequences[i] for i in train_idx]
            X_fold_val = [X_sequences[i] for i in val_idx]
            y_fold_val = [y_sequences[i] for i in val_idx]

            # Create and train a new model with the same initial weights
            model = NetworkTrafficRNN(
                input_size=X_sequences[0].shape[1],
                hidden_size=hyper_param_conf['hidden_size'],
                num_layers=hyper_param_conf['num_layers'],
                num_classes=4,
                cell_type=rnn_cell,
                dropout=hyper_param_conf['dropout'],
                hyperparams=hyper_param_conf
            )
            model.load_state_dict(torch.load('initial_rnn_vnf_model.pt'))

            # Train on this fold
            model.train_rnn(X_fold_train, y_fold_train, X_fold_val, y_fold_val)

            # Evaluate on validation set
            _, f1_score, _ = model.evaluate_rnn(X_fold_val, y_fold_val)
            avg_f1_score += f1_score

            if is_verbose:
                print(f"Fold {fold_idx+1} F1 Score: {f1_score:.4f}")

            # Early stop after one fold for faster iteration during development
            # Remove this in production for full k-fold CV
            # break

        # Calculate average F1 score across folds
        avg_f1_score /= fold_count

        if avg_f1_score > best_score:
            best_score = avg_f1_score
            best_params = hyper_param_conf

        if is_verbose:
            print(f"Average F1 Score: {avg_f1_score:.4f}")

    end_time = time.time()

    if is_verbose:
        print(f"Cross validation time: {end_time - start_time:.2f} seconds")
        print("Best Parameters based on Grid Search:")
        print(best_params)
        print(f"Best F1 Score: {best_score:.4f}")

    return best_params


def get_sequences(dataset_id=-1):
    datasets = {
        -1: "vALL",
        0: "vIDS",
        1: "vDNS",
        2: "vLB",
        3: "vProxy",
        4: "vRouter_vFW"
    }
    output_filename_base = datasets.get(dataset_id, "vALL")
    path = Path(f"data/VNFCyberData/{output_filename_base}_sequences.npz")

    if not path.exists():
        return None

    loaded_sequences = np.load(path, allow_pickle=False)
    return {
        "X_train": loaded_sequences["X_train"],
        "y_train": loaded_sequences["y_train"],
        "X_val": loaded_sequences["X_val"],
        "y_val": loaded_sequences["y_val"],
        "X_test": loaded_sequences["X_test"],
        "y_test": loaded_sequences["y_test"]
    }


def prepare_sequences(
    dataset_ids=None,
    window_size_msec=20,
    stride_size_msec=50,
    benign_sampling_ratio=1.0
):
    def _create_sequences_from_dataframe(
        df, dataset_ids=None,
        benign_sampling_ratio=1.0,
        output_sequence_filename_base="vALL"
    ):
        X, y = preprocess_df(
            df, use_diversity_index=False,
            benign_sampling_ratio=benign_sampling_ratio
        )
        X = X.drop(columns=['start_time'])
        X = X.drop(columns=['stop_time'])
        X = X.drop(columns=['source_country_normalized'])
        X = X.drop(columns=['destination_country_normalized'])
        X = X.drop(columns=['source_mac_normalized'])
        X = X.drop(columns=['destination_asn_normalized'])
        X = X.drop(columns=['payload_source_utf8_normalized'])
        X = X.drop(columns=['payload_destination_utf8_normalized'])
        # X = X.drop(columns=['protocols_normalized'])
        # X = X.drop(columns=['version_normalized'])
        X = X.drop(columns=['uri_normalized'])
        X = X.drop(columns=['host_normalized'])
        X = X.drop(columns=['hostname_normalized'])
        X = X.drop(columns=['alt_name_normalized'])
        X = X.drop(columns=['geo_normalized'])
        X = X.drop(columns=['tcp_flag_urg'])
        X = X.drop(columns=['type'])

        # Drop IP parts and port labels to reduce bias
        for i in range(1, 9):
            X = X.drop(columns=[f'source_ip_part{i}'])
            X = X.drop(columns=[f'destination_ip_part{i}'])
        X = X.drop(columns=['source_port_label_normalized'])
        X = X.drop(columns=['destination_port_label_normalized'])

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        feature_mean = X.mean(axis=0)  # Series of length n_features
        feature_std = X.std(axis=0) + 1e-6  # Series of length n_features
        X_norm = (X - feature_mean) / feature_std  # DataFrame, same shape as X
        X_norm = np.clip(X_norm, -5, 5)

        # Combine features and labels into a single DataFrame to allow dataset filtering
        df_combined = X.copy()
        df_combined['label'] = y.values if hasattr(y, "values") else np.array(y)

        # If dataset_ids provided, filter rows by dataset_id column (if present)
        if dataset_ids is not None and 'dataset_id' in df_combined.columns:
            print(f"Filtering datasets to IDs: {dataset_ids}")
            print(df_combined['dataset_id'])
            df_combined = df_combined[df_combined['dataset_id'].isin(dataset_ids)].reset_index(drop=True)

        # Split back into features and labels
        y = df_combined['label']
        X_norm = df_combined.drop(columns=['label'])
        # Drop dataset_id column if present before splitting back
        if 'dataset_id' in X_norm.columns:
            X_norm = X_norm.drop(columns=['dataset_id'])

        print(f"X_norm shape: {X_norm.shape}")
        print(f"y shape: {y.shape}")

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_norm, y, test_size=TEST_RATIO,
            stratify=y, random_state=RANDOM_STATE
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=VALIDATION_RATIO / (1 - TEST_RATIO),
            stratify=y_train_val, random_state=RANDOM_STATE
        )

        training_ds = pd.DataFrame(X_train)
        training_ds['label'] = y_train
        if 'timestamp' not in training_ds.columns:
            training_ds['timestamp'] = np.arange(len(training_ds))
        # X_train_seq, y_train_seq = create_time_window_sequences(training_ds, window_size_msec, stride_size_msec)
        X_train_seq, y_train_seq = create_fixed_length_sequences(training_ds, sequence_length=1, stride=1,
                                                                 max_sequences=10000000, balance_classes=False)
        # # create a CSV file for training_ds using X_train_seq and y_train_seq
        # print(X_train_seq)
        # training_ds_seq = pd.DataFrame(X_train_seq)
        # training_ds_seq['label'] = y_train_seq
        # training_ds_seq.to_csv(f"{output_filename_base}_train_seq.csv", index=False)

        validation_ds = pd.DataFrame(X_val)
        validation_ds['label'] = y_val
        if 'timestamp' not in validation_ds.columns:
            validation_ds['timestamp'] = np.arange(len(validation_ds))
        # X_val_seq, y_val_seq = create_time_window_sequences(validation_ds, window_size_msec, stride_size_msec)
        X_val_seq, y_val_seq = create_fixed_length_sequences(validation_ds, sequence_length=1, stride=1,
                                                             max_sequences=10000000, balance_classes=False)
        # # create a CSV file for val_ds using X_val_seq and y_val_seq
        # val_ds_seq = pd.DataFrame(X_val_seq)
        # val_ds_seq['label'] = y_val_seq
        # val_ds_seq.to_csv(f"{output_filename_base}_val_seq.csv", index=False)

        test_ds = pd.DataFrame(X_test)
        test_ds['label'] = y_test
        if 'timestamp' not in test_ds.columns:
            test_ds['timestamp'] = np.arange(len(test_ds))
        # X_test_seq, y_test_seq = create_time_window_sequences(test_ds, window_size_msec, stride_size_msec)
        X_test_seq, y_test_seq = create_fixed_length_sequences(test_ds, sequence_length=1, stride=1,
                                                               max_sequences=10000000, balance_classes=False)
        # # create a CSV file for test_ds using X_test_seq and y_test_seq
        # test_ds_seq = pd.DataFrame(X_test_seq)
        # test_ds_seq['label'] = y_test_seq
        # test_ds_seq.to_csv(f"{output_filename_base}_test_seq.csv", index=False)

        np.savez_compressed(
            f"data/VNFCyberData/{output_sequence_filename_base}_sequences.npz",
            X_train=X_train_seq,
            y_train=y_train_seq,
            X_val=X_val_seq,
            y_val=y_val_seq,
            X_test=X_test_seq,
            y_test=y_test_seq
        )

        return {
            "X_train": X_train_seq,
            "y_train": y_train_seq,
            "X_val": X_val_seq,
            "y_val": y_val_seq,
            "X_test": X_test_seq,
            "y_test": y_test_seq,
            "X_norm": X_norm.drop(columns=['timestamp']),
        }

    datasets = {
        0: "vIDS",
        1: "vDNS",
        2: "vLB",
        3: "vProxy",
        4: "vRouter_vFW"
    }

    output_sequence_filename_base = "vALL"
    if dataset_ids is not None:
        output_sequence_filename_base = "_".join(
            [datasets[ind] for ind in dataset_ids]
        )

    all_df = pd.DataFrame()
    for ds_id in datasets.keys():
        ds_name = datasets.get(ds_id)
        ds_df = load_data(Path(f"data/VNFCyberData/{ds_name}.csv"))
        ds_df['dataset_id'] = ds_id
        all_df = pd.concat([all_df, ds_df], ignore_index=True)

    return _create_sequences_from_dataframe(
        all_df,
        dataset_ids=dataset_ids,
        benign_sampling_ratio=benign_sampling_ratio,
        output_sequence_filename_base=output_sequence_filename_base
    )


def initialize_model():
    """Initialize the model with default parameters"""
    return NetworkTrafficRNN(
        input_size=72,  # Example input size, adjust as needed
        hidden_size=128,
        num_layers=2,
        num_classes=4,  # Example number of classes, adjust as needed
        cell_type="LSTM",
        dropout=0.05,
        hyperparams=None
    )


def train_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq):
    """Train the RNN model"""
    if len(X_train_seq) == 0:
        raise ValueError("Empty training data")

    return train_rnn_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, rnn_cell="LSTM", ext_model=model)


def evaluate_model(model, X_test_seq, y_test_seq):
    """Evaluate the RNN model"""
    if len(X_test_seq) == 0:
        raise ValueError("Empty test data")

    return evaluate_rnn_model(model, X_test_seq, y_test_seq, ext_model=model)

if __name__ == '__main__':
    #RANDOM_STATE = 42

    # dataset_indices = [1, 3, 5]#, 5, 6, 7, 8, 9]
    # df = pd.DataFrame()
    # for ds_index in dataset_indices:
    #     ds_df = load_data(Path(f"data/sessions_{ds_index}_vDNS.csv"))
    #     df = pd.concat([df, ds_df], ignore_index=True)
    dataset_names = ["vIDS.csv", "vDNS.csv", "vLB.csv", "vProxy.csv", "vRouter_vFW.csv"]
    df = pd.DataFrame()
    for ds_name in dataset_names:
        ds_df = load_data(Path(f"data/VNFCyberData/{ds_name}"))
        df = pd.concat([df, ds_df], ignore_index=True)

    print(df.groupby(['label']).size())
    X, y = preprocess_df(df, use_diversity_index=False)

    X = X.drop(columns=['source_mac_normalized'])
    X = X.drop(columns=['destination_asn_normalized'])
    X = X.drop(columns=['protocols_normalized'])
    X = X.drop(columns=['version_normalized'])
    X = X.drop(columns=['uri_normalized'])
    X = X.drop(columns=['host_normalized'])
    X = X.drop(columns=['hostname_normalized'])
    X = X.drop(columns=['alt_name_normalized'])
    X = X.drop(columns=['geo_normalized'])

    for i in range(1, 9):
        X = X.drop(columns=[f'source_ip_part{i}'])
        X = X.drop(columns=[f'destination_ip_part{i}'])
    X = X.drop(columns=['source_port_label_normalized'])
    X = X.drop(columns=['destination_port_label_normalized'])

    feature_mean = X.mean(axis=0)  # Series of length n_features
    feature_std = X.std(axis=0) + 1e-6  # Series of length n_features
    X_norm = (X - feature_mean) / feature_std  # DataFrame, same shape as X

    print(f"Data shape X: {X.shape}, Labels shape: {y.shape}")
    print(f"Data shape X_norm: {X_norm.shape}, Labels shape: {y.shape}")
    print("Continue....")

    for _ in range(TRIALS):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_norm, y, test_size=TEST_RATIO,
            stratify=y, #random_state=RANDOM_STATE
        )
        # rf = train_random_forest_with_k_fold_cv(
        #     X_train_val, y_train_val, is_verbose=True
        # )
        # cm = test_random_forest(rf, X_test, y_test)
        # print(cm)

        # rf = train_random_forest(
        #     X_train_val, y_train_val,
        #     max_depth=6, n_estimators=100
        # )
        # cm = test_random_forest(rf, X_test, y_test)
        # print(report_cm_results(cm))

        # svm = train_svm(X_train, y_train, dual='auto')
        # cm = test_svm(svm, X_test, y_test)
        # print(report_cm_results(cm))

        # svc = train_svm_with_k_fold_cv(
        #     X_train_val, y_train_val, is_verbose=True
        # )
        # cm = test_svm(svc, X_test, y_test)
        # print(report_cm_results(cm))

        # Sequence creation for RNN
        window_size = 1  # seconds
        stride_size = 0.5   # seconds

        # Create a data frame with features and label that includes timestamp
        df_with_timestamp = pd.DataFrame(X_norm)
        df_with_timestamp['label'] = y

        # Make sure timestamp is in the right format
        # In VNF data, we can use start_time or combine start_time and stop_time
        if 'timestamp' not in df_with_timestamp.columns:
            if 'start_time' in df.columns:
                # Use start_time from original dataframe if available
                df_with_timestamp['timestamp'] = df['start_time'].astype(int)
            else:
                # Create a synthetic timestamp for demonstration
                print("Creating synthetic timestamps as real timestamps not found")
                df_with_timestamp['timestamp'] = np.arange(len(df_with_timestamp))

        # Now create sequences with the properly formatted dataframe
        X_sequences, y_sequences = create_fixed_length_sequences(df_with_timestamp, 100, 15, max_sequences=1000000)
        print(X_sequences[:10])
        print(X_sequences[-10:])

        # Train RNN with K-Fold CV
        best_rnn_params = train_rnn_with_k_fold_cv(X_sequences, y_sequences, rnn_cell="RNN", is_verbose=True)

        # For the final model evaluation, we need to ensure X_test is in the right format
        # First normalize the test data the same way we did with training data
        # Convert test dataframe to sequence format
        df_test_with_timestamp = pd.DataFrame(X_test)
        df_test_with_timestamp['label'] = y_test

        if 'timestamp' not in df_test_with_timestamp.columns:
            # Create synthetic timestamps for test data
            print("Creating synthetic timestamps for test data")
            df_test_with_timestamp['timestamp'] = np.arange(len(df_test_with_timestamp))

        # Create sequences from test data
        X_test_sequences, y_test_sequences = create_time_window_sequences(
            df_test_with_timestamp,
            window_size,
            stride_size,
            max_sequences=100  # Limit test sequences for performance
        )

        # If we have no test sequences, we can't evaluate
        if len(X_test_sequences) == 0:
            print("Warning: No test sequences could be created. Cannot evaluate the model.")
            continue

        print(f"Created {len(X_test_sequences)} test sequences")

        # Train final model with the best parameters found
        final_rnn_model, _, _ = train_rnn_model(
            X_sequences, y_sequences,  # Use all training sequences
            X_test_sequences, y_test_sequences,  # Use sequences created from test data
            rnn_cell="RNN",
            hyperparams=best_rnn_params
        )

        # Evaluate final model
        _, f1_score, report = evaluate_rnn_model(final_rnn_model, X_test_sequences, y_test_sequences)
        print(f"Final Model F1 Score: {f1_score:.4f}")
        print(report)
