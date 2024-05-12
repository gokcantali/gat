import os
from dataclasses import dataclass
from typing import Any, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from gat.converter import convert_to_graph
from gat.data_models import Metrics
from gat.model import GAT
from gat.preprocesser import preprocess_df, preprocess_X, preprocess_y

TEST_SIZE = 0.25
RANDOM_STATE = 42
NUM_RUNS = 1  # Number of runs for each configuration to average the metrics

@dataclass
class Config:
    optimizer: Any
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    hidden_dim: int
    dropout: float

@dataclass
class ConfigResults:
    config: Config
    metrics: List[Metrics]
    avg_val_accuracy: float
    avg_val_loss: float
    avg_train_loss: float
    avg_val_precision: float
    avg_val_recall: float
    avg_val_f1: float
    composite_score: float

def split_data():
    df = preprocess_df()
    X = preprocess_X(df)
    y = preprocess_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    train_data = convert_to_graph(X_train, y_train)
    test_data = convert_to_graph(X_test, y_test)
    return train_data, test_data, y_train, y_test

def initialize_model(train_data, y_train, config):
    model = GAT(
        optimizer=config.optimizer,
        num_features=train_data.num_features,
        num_classes=len(np.unique(y_train)),
        weight_decay=config.weight_decay,
        dropout=config.dropout,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        lr=config.lr,
        patience=config.patience
    )
    return model

def run_experiment(config):
    metrics_list = []
    for _ in range(NUM_RUNS):
        train_data, test_data, y_train, y_test = split_data()
        model = initialize_model(train_data, y_train, config)
        metrics: Metrics = model.train_model(train_data, test_data)
        metrics_list.append(metrics)

    avg_val_accuracy = np.mean([m.val_accuracy[-1] for m in metrics_list])
    avg_val_loss = np.mean([m.val_loss[-1] for m in metrics_list])
    avg_train_loss = np.mean([m.train_loss[-1] for m in metrics_list])
    avg_val_precision = np.mean([m.val_precision[-1] for m in metrics_list])
    avg_val_recall = np.mean([m.val_recall[-1] for m in metrics_list])
    avg_val_f1 = np.mean([m.val_f1[-1] for m in metrics_list])

    # Normalize metrics (e.g., between 0 and 1)
    max_accuracy = 1.0
    max_precision = 1.0
    max_recall = 1.0
    max_f1 = 1.0

    norm_accuracy = avg_val_accuracy / max_accuracy
    norm_precision = avg_val_precision / max_precision
    norm_recall = avg_val_recall / max_recall
    norm_f1 = avg_val_f1 / max_f1

    # Define weights for each metric
    weight_accuracy = 0.25
    weight_precision = 0.25
    weight_recall = 0.25
    weight_f1 = 0.25

    composite_score = (
        weight_accuracy * norm_accuracy +
        weight_precision * norm_precision +
        weight_recall * norm_recall +
        weight_f1 * norm_f1
    )

    return avg_val_accuracy, avg_val_loss, avg_train_loss, avg_val_precision, avg_val_recall, avg_val_f1, composite_score, metrics_list

def grid_search(configurations):
    best_config = None
    best_composite_score = 0
    all_results = []

    for config in configurations:
        print(f"Running experiment with config: {config}")
        avg_val_accuracy, avg_val_loss, avg_train_loss, avg_val_precision, avg_val_recall, avg_val_f1, composite_score, metrics_list = run_experiment(config)
        config_result = ConfigResults(
            config=config,
            metrics=metrics_list,
            avg_val_accuracy=avg_val_accuracy,
            avg_val_loss=avg_val_loss,
            avg_train_loss=avg_train_loss,
            avg_val_precision=avg_val_precision,
            avg_val_recall=avg_val_recall,
            avg_val_f1=avg_val_f1,
            composite_score=composite_score
        )
        all_results.append(config_result)

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_config = config

    return best_config, best_composite_score, all_results

def main():
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Define grid of hyperparameters to search
    optimizers = [torch.optim.AdamW]
    lrs = [0.04]
    weight_decays = [5e-4]
    epochs_list = [30]
    patiences = [3]
    hidden_dims = [32]
    dropouts = [0.4]

    configurations = [
        Config(optimizer, lr, weight_decay, epochs, patience, hidden_dim, dropout)
        for optimizer in optimizers
        for lr in lrs
        for weight_decay in weight_decays
        for epochs in epochs_list
        for patience in patiences
        for hidden_dim in hidden_dims
        for dropout in dropouts
    ]

    best_config, best_composite_score, all_results = grid_search(configurations)
    print(f"Best config: {best_config} with composite score: {best_composite_score}")

    # Save results for further analysis
    with open("./results/all_results.txt", "w") as f:
        for result in all_results:
            f.write(f"Config: {result.config}, "
                    f"Avg Val Accuracy: {result.avg_val_accuracy}, "
                    f"Avg Val Loss: {result.avg_val_loss}, "
                    f"Avg Train Loss: {result.avg_train_loss}, "
                    f"Avg Val Precision: {result.avg_val_precision}, "
                    f"Avg Val Recall: {result.avg_val_recall}, "
                    f"Avg Val F1: {result.avg_val_f1}, "
                    f"Composite Score: {result.composite_score}\n")

if __name__ == "__main__":
    main()
