import os
import random
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
NUM_SAMPLES = 10  # Number of random configurations to sample

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

def random_search(config_ranges, num_samples):
    best_config = None
    best_composite_score = 0
    all_results = []

    for _ in range(num_samples):
        config = Config(
            optimizer=random.choice(config_ranges["optimizers"]),
            lr=random.uniform(*config_ranges["lr"]),
            weight_decay=random.uniform(*config_ranges["weight_decay"]),
            epochs=random.choice(config_ranges["epochs"]),
            patience=random.choice(config_ranges["patience"]),
            hidden_dim=random.choice(config_ranges["hidden_dim"]),
            dropout=random.uniform(*config_ranges["dropout"])
        )
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

    # Define ranges of hyperparameters to search
    config_ranges = {
        "optimizers": [torch.optim.AdamW],
        "lr": (0.03, 0.05),  # As a tuple for random.uniform
        "weight_decay": (0.0004, 0.0006),  # As a tuple for random.uniform
        "epochs": [20, 30, 40],
        "patience": [3, 5, 7],
        "hidden_dim": [25, 32, 39],
        "dropout": (0.3, 0.5)  # As a tuple for random.uniform
    }

    best_config, best_composite_score, all_results = random_search(config_ranges, NUM_SAMPLES)
    print(f"Best config: {best_config} with composite score: {best_composite_score}")

    # Save results for further analysis
    with open("./results/all_results_random_search.txt", "w") as f:
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
