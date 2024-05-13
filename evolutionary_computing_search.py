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
POPULATION_SIZE = 10
NUM_GENERATIONS = 5
MUTATION_RATE = 0.1
GLOBAL_X = None
GLOBAL_Y = None

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
    global GLOBAL_X, GLOBAL_Y
    if GLOBAL_X is None or GLOBAL_Y is None:
        df = preprocess_df()
        GLOBAL_X = preprocess_X(df)
        GLOBAL_Y = preprocess_y(df)
    X = GLOBAL_X
    y = GLOBAL_Y
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

def create_initial_population(size, config_ranges):
    population = []
    for _ in range(size):
        config = Config(
            optimizer=random.choice(config_ranges["optimizers"]),
            lr=random.choice(config_ranges["lr"]),  # Use random.choice for discrete values
            weight_decay=random.uniform(*config_ranges["weight_decay"]),  # Use random.uniform for continuous range
            epochs=random.choice(config_ranges["epochs"]),
            patience=random.choice(config_ranges["patience"]),
            hidden_dim=random.choice(config_ranges["hidden_dim"]),
            dropout=random.choice(config_ranges["dropout"])  # Use random.choice for discrete values
        )
        population.append(config)
    return population

def mutate_config(config, config_ranges):
    if random.random() < MUTATION_RATE:
        config.lr = random.choice(config_ranges["lr"])  # Use random.choice for discrete values
    if random.random() < MUTATION_RATE:
        config.weight_decay = random.uniform(*config_ranges["weight_decay"])  # Use random.uniform for continuous range
    if random.random() < MUTATION_RATE:
        config.epochs = random.choice(config_ranges["epochs"])
    if random.random() < MUTATION_RATE:
        config.patience = random.choice(config_ranges["patience"])
    if random.random() < MUTATION_RATE:
        config.hidden_dim = random.choice(config_ranges["hidden_dim"])
    if random.random() < MUTATION_RATE:
        config.dropout = random.choice(config_ranges["dropout"])  # Use random.choice for discrete values
    return config

def crossover_configs(config1, config2):
    child1 = Config(
        optimizer=random.choice([config1.optimizer, config2.optimizer]),
        lr=random.choice([config1.lr, config2.lr]),
        weight_decay=random.choice([config1.weight_decay, config2.weight_decay]),
        epochs=random.choice([config1.epochs, config2.epochs]),
        patience=random.choice([config1.patience, config2.patience]),
        hidden_dim=random.choice([config1.hidden_dim, config2.hidden_dim]),
        dropout=random.choice([config1.dropout, config2.dropout])
    )
    child2 = Config(
        optimizer=random.choice([config1.optimizer, config2.optimizer]),
        lr=random.choice([config1.lr, config2.lr]),
        weight_decay=random.choice([config1.weight_decay, config2.weight_decay]),
        epochs=random.choice([config1.epochs, config2.epochs]),
        patience=random.choice([config1.patience, config2.patience]),
        hidden_dim=random.choice([config1.hidden_dim, config2.hidden_dim]),
        dropout=random.choice([config1.dropout, config2.dropout])
    )
    return child1, child2

def evolve_population(population, config_ranges):
    new_population = []
    population_scores = []

    for config in population:
        avg_val_accuracy, avg_val_loss, avg_train_loss, avg_val_precision, avg_val_recall, avg_val_f1, composite_score, metrics_list = run_experiment(config)
        population_scores.append((composite_score, config))

    # Sort population by composite score
    population_scores.sort(reverse=True, key=lambda x: x[0])
    top_performers = population_scores[:POPULATION_SIZE // 2]

    for score, config in top_performers:
        new_population.append(config)
        child1, child2 = crossover_configs(config, random.choice(top_performers)[1])
        new_population.extend([mutate_config(child1, config_ranges), mutate_config(child2, config_ranges)])

    return new_population

def evolutionary_search(config_ranges):
    population = create_initial_population(POPULATION_SIZE, config_ranges)
    best_config = None
    best_composite_score = 0

    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation + 1}")
        population = evolve_population(population, config_ranges)

        for config in population:
            avg_val_accuracy, avg_val_loss, avg_train_loss, avg_val_precision, avg_val_recall, avg_val_f1, composite_score, metrics_list = run_experiment(config)
            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_config = config

    return best_config, best_composite_score

def main():
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Define ranges of hyperparameters to search
    config_ranges = {
        "optimizers": [torch.optim.AdamW],
        "lr": [0.039, 0.04, 0.041, 0.042, 0.0425],
        "weight_decay": (0.00046, 0.00052),
        "epochs": [30, 40],
        "patience": [5],
        "hidden_dim": [28, 30, 32, 34, 36],
        "dropout": [0.35, 0.375, 0.4, 0.425, 0.45]
    }

    best_config, best_composite_score = evolutionary_search(config_ranges)
    print(f"Best config: {best_config} with composite score: {best_composite_score}")

    # Save results for further analysis
    with open("./results/best_result_evolutionary.txt", "w") as f:
        f.write(f"Best Config: {best_config}\nComposite Score: {best_composite_score}\n")

if __name__ == "__main__":
    main()
