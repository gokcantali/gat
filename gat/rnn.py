import sys
import warnings
from collections import Counter, OrderedDict
from typing import List

from sklearn.metrics import f1_score, classification_report

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import classification_report, f1_score as calculate_f1_score
from torch.utils.data import DataLoader, TensorDataset

from gat.load_data import load_data
from gat.preprocesser import preprocess_X, preprocess_y, \
    construct_port_scan_label
from gat.rnn_preprocessor import create_sequences, create_sequences_based_on_session, \
    create_time_window_sequences, process_timestamps


def subset(lst, indices):
    """Return a new list with elements lst[i] for every i in indices (np.ndarray)."""
    return [lst[int(i)] for i in indices]


def pad_collate_fn(batch):
    """
    Turns [(seq_i, label_i, len_i), ...] into
    padded_seq_tensor [B, max_len, F], labels [B], lengths [B].
    """
    seqs, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)

    padded_seqs = pad_sequence(seqs, batch_first=True)  # zero-pads

    return padded_seqs, torch.tensor(labels), lengths


class SequenceDataset(torch.utils.data.Dataset):
    """
    Keeps variable-length sequences (already normalised!) and their labels.
    """
    def __init__(self, sequences, labels):
        assert len(sequences) == len(labels)
        self.seqs   = sequences
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        # do NOT pad here – just hand the raw tensor back
        seq   = torch.tensor(self.seqs[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label, seq.shape[0]      # length used later


class NetworkTrafficRNN(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout,
        cell_type="LSTM",
        is_bidirectional=False,
        hyperparams=None
    ):
        super(NetworkTrafficRNN, self).__init__()
        if hyperparams is None:
            hyperparams = {}
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN layer
        cell_type = cell_type.upper()
        rnn_cls = {"LSTM": torch.nn.LSTM, "GRU": torch.nn.GRU, "RNN": torch.nn.RNN}[cell_type]

        self.rnn = rnn_cls(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=is_bidirectional
        )

        # Fully connected layer
        self.fc = torch.nn.Linear(
            hidden_size * (2 if is_bidirectional else 1),
            num_classes
        )

        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)

        # Computation Device
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_bidirectional = is_bidirectional

        # Set hyperparameters
        if type(hyperparams) is dict:
            self.batch_size = hyperparams.get('batch_size', 32)
            self.learning_rate = hyperparams.get('learning_rate', 0.001)
            self.num_epochs = hyperparams.get('num_epochs', 10)
            self.patience = hyperparams.get('patience', 3)

    def forward(self, x, lengths):
        # Initialize hidden state
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.rnn(packed)

        # hidden is:
        #   • (h_n, c_n) for LSTM
        #   • h_n        for GRU / RNN
        if isinstance(hidden, tuple):  # LSTM
            h_n = hidden[0]
        else:  # GRU / RNN
            h_n = hidden

        last = h_n[-1]  # [B, hidden * directions]
        out = self.fc(self.dropout(last))
        return out

    def train_rnn(self, X_train, y_train, X_val, y_val):
        # Create DataLoader
        train_loader = DataLoader(
            SequenceDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=pad_collate_fn
        )

        val_loader = DataLoader(
            SequenceDataset(X_val, y_val),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=pad_collate_fn
        )

        # Loss and optimizer
        # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        best_val_loss = float('inf')
        best_f1_score = 0.0
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            for padded_X, labels, lengths in train_loader:
                # inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self(padded_X, lengths)
                loss = criterion(logits, labels)
                #outputs = model(inputs)
                #loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * labels.size(0)

            # Validation phase
            self.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            predicted_list = []
            correct_list = []
            with torch.no_grad():
                for padded_X, labels, lengths in val_loader:
                    # inputs, labels = inputs.to(self.device), labels.to(self.device)
                    logits = self(padded_X, lengths)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * labels.size(0)

                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    predicted_list.append(predicted.cpu())
                    correct_list.append(labels.cpu())

            # Calculate metrics
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_f1_score = calculate_f1_score(correct_list, predicted_list, average='weighted'),
                patience_counter = 0
                torch.save(self.state_dict(), '../rnn_best_model.pth')  # Save best model
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    # print("Early stopping triggered!")
                    self.load_state_dict(torch.load("../rnn_best_model.pth"))
                    break

        self.load_state_dict(torch.load("../rnn_best_model.pth"))
        return best_val_loss, best_f1_score

    def evaluate_rnn(self, X, y):
        self.eval()

        criterion = torch.nn.CrossEntropyLoss()
        test_loader = DataLoader(
            SequenceDataset(X, y),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=pad_collate_fn,
        )

        prediction_list, label_list = [], []
        test_loss = 0.0
        with torch.no_grad():
            for padded_X, labels, lengths in test_loader:
                logits = self(padded_X, lengths)
                prediction_list.append(torch.argmax(logits, 1).cpu())
                label_list.append(labels.cpu())

                loss = criterion(logits, labels)
                test_loss += loss.item() * labels.size(0)

        y_true = torch.cat(label_list)
        y_pred = torch.cat(prediction_list)

        return (
            test_loss / len(test_loader.dataset),
            calculate_f1_score(y_true, y_pred, average='weighted'),
            classification_report(y_true, y_pred, output_dict=True)
        )

    def set_parameters(self, parameters: List[np.ndarray], config, is_evaluate=False):
        if config is None:
            config = {}
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

        model_file_name = self._construct_model_file_name(config)

        if is_evaluate is False:
            torch.save(state_dict, model_file_name)

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    @staticmethod
    def _construct_model_file_name(config):
        total_rounds = config.get("total_rounds", 20)
        model_file_name = f"best_RNN_model_FL_{total_rounds}_"

        cf_method = config.get("cf_method", "NON_CF")
        model_file_name += cf_method

        trial = config.get("trial", "First")
        model_file_name += f"-{trial}.pt"

        return model_file_name

def create_hyper_param_space():
    """ constructs an array of hyperparameter configuration based on the
        cartesian product of the hyperparameter values """
    from itertools import product

    hyper_param_space = {
        "learning_rate": [0.001],# 0.01, 0.05],
        "batch_size": [32, 64],# 128],
        "num_epochs": [20],#[10, 20],# 30],
        "patience": [3],# 5, 7],
        "num_layers": [2,],# 3],
        "hidden_size": [64, 128],# 256],
        "dropout": [0.05],# 0.1, 0.2],
    }

    # Create a list of all combinations of hyperparameters
    # with a list of dictionaries for each combination
    hyper_param_combs = []
    for values in product(*hyper_param_space.values()):
        hyper_param_combination = {}
        for i, key in enumerate(hyper_param_space.keys()):
            hyper_param_combination[key] = values[i]
        hyper_param_combs.append(hyper_param_combination)

    return hyper_param_combs


def create_hyper_param_space_for_SVM():
    """ constructs an array of hyperparameter configuration based on the
        cartesian product of the hyperparameter values """
    from itertools import product

    hyper_param_space = {
        "kernel": ["rbf", "linear"],
        "C": [0.01, 0.1, 1],
        "tol": [0.001, 0.005], # tolerance for stopping criteria
    }

    # Create a list of all combinations of hyperparameters
    # with a list of dictionaries for each combination
    hyper_param_combs = []
    for values in product(*hyper_param_space.values()):
        hyper_param_combination = {}
        for i, key in enumerate(hyper_param_space.keys()):
            hyper_param_combination[key] = values[i]
        hyper_param_combs.append(hyper_param_combination)

    return hyper_param_combs


def create_hyper_param_space_for_random_forest():
    """ constructs an array of hyperparameter configuration based on the
        cartesian product of the hyperparameter values """
    from itertools import product

    hyper_param_space = {
        "max_depth": [5, 10, 20],#, 15, 20, 50, 100],
        "n_estimators": [50, 100, 150],# 150, 250, 500],
        "max_leaf_nodes": [10, 20, 30],# 30, 40, 50]
    }

    # Create a list of all combinations of hyperparameters
    # with a list of dictionaries for each combination
    hyper_param_combs = []
    for values in product(*hyper_param_space.values()):
        hyper_param_combination = {}
        for i, key in enumerate(hyper_param_space.keys()):
            hyper_param_combination[key] = values[i]
        hyper_param_combs.append(hyper_param_combination)

    return hyper_param_combs


def benchmark_model(classifier="RF"):
    """
    Benchmark random forest or SVM classifier on the preprocessed data.
    This function is used to compare the performance of the RNN model
    with a baseline random forest model.
    """
    from gat.load_data import load_data
    from gat.preprocesser import preprocess_X, preprocess_y, construct_port_scan_label

    df = load_data(
        "../data/subsample/traces-benign0.01-dos0.10-port0.10-zap1.00.csv",
    )
    df = df.sort_values(by=['timestamp'])
    df = construct_port_scan_label(df, use_diversity_index=False)
    df = df.reset_index(drop=True)
    df["is_anomaly"] = df["is_anomaly"].replace({"True": 1, "False": 0}).astype(int)


    X = preprocess_X(
        df, use_diversity_index=False, keep_timestamp=False
    )
    y = preprocess_y(df)

    # Normalization is a must here!
    mean = X.values.mean(axis=(0, 1))
    std = X.values.std(axis=(0, 1))
    X_norm = (X.values - mean) / (std + 1e-6)

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in outer_cv.split(X_norm, y):
        X_train_val, X_test = subset(X_norm, train_index), subset(X_norm, test_index)
        y_train_val, y_test = subset(y, train_index), subset(y, test_index)

        if classifier == "RF":
            hyper_param_combinations = create_hyper_param_space_for_random_forest()
        elif classifier == "SVM":
            hyper_param_combinations = create_hyper_param_space_for_SVM()

        best_score = 0.0
        best_params = None
        for hyper_param_conf in hyper_param_combinations:
            # Create a new model instance for each hyperparameter configuration
            if classifier == "SVM":
                model = SVC(**hyper_param_conf, class_weight="balanced")
            elif classifier == "RF":
                model = RandomForestClassifier(**hyper_param_conf, class_weight="balanced")

            # Create the inner cross-validation for hyperparameter tuning
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            avg_f1_score_per_fold = 0.0
            for inner_train_val_index, inner_test_index in inner_cv.split(X_train_val, y_train_val):
                X_inner_train_val, X_inner_test = subset(X_train_val, inner_train_val_index), subset(X_train_val, inner_test_index)
                y_inner_train_val, y_inner_test = subset(y_train_val, inner_train_val_index), subset(y_train_val, inner_test_index)

                # Train the model
                model.fit(
                    X_inner_train_val,
                    y_inner_train_val,
                )

                # Evaluate the model
                y_predict = model.predict(X_inner_test)
                # use scikit learn for f1 score

                rf_f1_score = f1_score(y_inner_test, y_predict, average='weighted')
                avg_f1_score_per_fold += rf_f1_score / 3

            # Save the best model
            if avg_f1_score_per_fold > best_score:
                best_score = avg_f1_score_per_fold
                best_params = hyper_param_conf

            print("[DONE] Hyper param conf: ", hyper_param_conf)

        print("Best F1 Score: ", best_score)
        print("Best params: ", best_params)

        X_outer_train, X_outer_val, y_outer_train, y_outer_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, shuffle=True,
            stratify=y_train_val, random_state=42
        )

        # Retrain the model on the best config
        model = None
        if classifier == "SVM":
            model = SVC(**best_params, class_weight="balanced")
        elif classifier == "RF":
            model = RandomForestClassifier(**best_params, class_weight="balanced")
        model.fit(
            X_outer_train,
            y_outer_train,
        )

        # Evaluate the model on the outer test set
        y_pred = model.predict(X_test)
        rf_f1_score = f1_score(y_test, y_pred, average='weighted')
        rf_clas_report = classification_report(y_test, y_pred, output_dict=True)
        print("Outer Classification Report")
        print(rf_clas_report)


if __name__ == '__main__':
    window_seconds, rnn_cell = int(sys.argv[1]), sys.argv[2]
    df = load_data(
        "../data/subsample/traces-benign0.01-dos0.10-port0.10-zap1.00.csv",
    )
    df = df.sort_values(by=['timestamp'])
    df = construct_port_scan_label(df, use_diversity_index=False)
    df["is_anomaly"] = df["is_anomaly"].replace({"True": 1, "False": 0}).astype(int)

    data = preprocess_X(
        df, use_diversity_index=False, keep_timestamp=True
    )
    data['label'] = preprocess_y(df)

    #data = process_timestamps(data, keep_timestamp=True)

    # sequence_length = 20
    # X, y = create_sequences(
    #     data, sequence_length
    # )

    # X, y = create_sequences_based_on_session(
    #     data
    # )
    # Normalization is a must here!
    # mean = X.mean(axis=(0, 1), keepdims=True)
    # std = X.std(axis=(0, 1), keepdims=True)
    # X_norm = (X - mean) / (std + 1e-6)

    X, y = create_time_window_sequences(
        data=data,
        window_seconds=window_seconds,
        label_reducer=lambda l: Counter(l).most_common(1)[0][0]
    )
    # # Normalization is a must here!
    all_frames = np.concatenate(X, axis=0)  # cat over time
    mean, std = all_frames.mean(0, keepdims=True), all_frames.std(0, keepdims=True)
    X_norm = [(seq - mean) / (std + 1e-6) for seq in X]

    # print("sequences")
    # print(X)

    # Construct train/test split with Stratified k-fold, not time series split
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in outer_cv.split(X_norm, y):
        X_train_val, X_test = subset(X_norm, train_index), subset(X_norm, test_index)
        y_train_val, y_test = subset(y, train_index), subset(y, test_index)

        hyper_param_combinations = create_hyper_param_space()
        best_score = 0.0
        best_params = None
        for hyper_param_conf in hyper_param_combinations:
            # Create a new model instance for each hyperparameter configuration
            model = NetworkTrafficRNN(
                input_size=X_train_val[0][0].size,
                hidden_size=hyper_param_conf['hidden_size'],
                num_layers=hyper_param_conf['num_layers'],
                num_classes=4,
                cell_type=rnn_cell,
                dropout=hyper_param_conf['dropout'],
                hyperparams=hyper_param_conf
            )
            # Save the initial model with default weights
            torch.save(model.state_dict(), '../initial_rnn_model.pt')

            # Create the inner cross-validation for hyperparameter tuning
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            avg_f1_score_per_fold = 0.0
            for inner_train_val_index, inner_test_index in inner_cv.split(X_train_val, y_train_val):
                X_inner_train_val, X_inner_test = subset(X_train_val, inner_train_val_index), subset(X_train_val, inner_test_index)
                y_inner_train_val, y_inner_test = subset(y_train_val, inner_train_val_index), subset(y_train_val, inner_test_index)

                X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
                    X_inner_train_val, y_inner_train_val, test_size=0.2, shuffle=True,
                    stratify=y_inner_train_val, random_state=42
                )

                # Load the initial model with default weights
                model.load_state_dict(torch.load('../initial_rnn_model.pt'))

                # Train the model
                model.train_rnn(
                    X_inner_train,
                    y_inner_train,
                    X_inner_val,
                    y_inner_val
                )

                # Evaluate the model
                _, f1_score, _ = model.evaluate_rnn(
                    X_inner_test,
                    y_inner_test
                )
                avg_f1_score_per_fold += f1_score / 3
                break

            # Save the best model
            if avg_f1_score_per_fold > best_score:
                best_score = avg_f1_score_per_fold
                best_params = hyper_param_conf

            print("[DONE] Hyper param conf: ", hyper_param_conf)

        print("Best F1 Score: ", best_score)
        print("Best params: ", best_params)

        X_outer_train, X_outer_val, y_outer_train, y_outer_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, shuffle=True,
            stratify=y_train_val, random_state=42
        )

        # Retrain the model on the best config
        model = NetworkTrafficRNN(
            input_size=X_train_val[0][0].size,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            num_classes=4,
            cell_type=rnn_cell,
            dropout=best_params['dropout'],
            hyperparams=best_params
        )
        model.train_rnn(
            X_outer_train,
            y_outer_train,
            X_outer_val,
            y_outer_val
        )

        # Evaluate the model on the outer test set
        _, _, classification_report = model.evaluate_rnn(
            X_test,
            y_test
        )
        print("Outer Classification Report")
        print(classification_report)

    # benchmark_model(classifier="SVM")
