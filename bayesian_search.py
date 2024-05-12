import os
from dataclasses import dataclass
from typing import Any, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

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

# Define the search space for Bayesian Optimization
space = [
    Categorical([torch.optim.AdamW, torch.optim.SGD, torch.optim.RMSprop], name="optimizer"),
    Real(0.001, 0.01, name="lr"),
    Real(5e-4, 1e-3, name="weight_decay"),
    Integer(30, 50, name="epochs"),
    Integer(3, 5, name="patience"),
    Integer(16, 64, name="hidden_dim"),
    Real(0.4, 0.6, name="dropout")
]

# Objective function to minimize
@use_named_args(space)
def objective(**params):
    config = Config(
        optimizer=params["optimizer"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        epochs=params["epochs"],
        patience=params["patience"],
        hidden_dim=params["hidden_dim"],
        dropout=params["dropout"]
    )
    avg_val_accuracy, avg_val_loss, avg_train_loss, avg_val_precision, avg_val_recall, avg_val_f1, composite_score, metrics_list = run_experiment(config)
    return -composite_score  # We minimize the negative composite score to maximize the actual composite score

def main():
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Perform Bayesian Optimization
    res = gp_minimize(objective, space, n_calls=50, random_state=RANDOM_STATE)

    best_params = res.x
    best_composite_score = -res.fun

    best_config = Config(
        optimizer=best_params[0],
        lr=best_params[1],
        weight_decay=best_params[2],
        epochs=best_params[3],
        patience=best_params[4],
        hidden_dim=best_params[5],
        dropout=best_params[6]
    )

    print(f"Best config: {best_config} with composite score: {best_composite_score}")

    # Save results for further analysis
    with open("./results/best_result.txt", "w") as f:
        f.write(f"Best Config: {best_config}\nComposite Score: {best_composite_score}\n")

if __name__ == "__main__":
    main()
