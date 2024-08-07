import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from gat.converter import convert_to_graph, create_tdg_graphs_using_window
from gat.model import GAT
from gat.gcn import GCN
from gat.preprocesser import preprocess_df, preprocess_X, preprocess_y

TEST_SIZE = 0.25
RANDOM_STATE = 42

class Config:
    optimizer = torch.optim.AdamW
    lr = 0.0425
    weight_decay = 0.0004807430799298252
    epochs = 30
    patience = 5
    hidden_dim = 30
    dropout = 0.425

def split_data():
    print("Start feature engineering...")
    df = preprocess_df()
    print("Feature engineering done.")
    print("Start preprocessing...")
    X = preprocess_X(df)
    y = preprocess_y(df)
    print("Preprocessing done.")
    print("Start splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print("Splitting data done.")
    print("Start converting to graph...")
    train_data = convert_to_graph(X_train, y_train)
    test_data = convert_to_graph(X_test, y_test)
    print("Converting to graph done.")
    return train_data, test_data, y_train


def split_data_for_tdg():
    print("Start feature engineering...")
    df = preprocess_df()

    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df['is_anomaly'], random_state=RANDOM_STATE)
    print("Splitting data done.")
    print("Start converting to graph...")
    train_data = create_tdg_graphs_using_window(df_train)
    test_data = create_tdg_graphs_using_window(df_test)
    print("Converting to graph done.")
    return train_data, test_data

def initialize_gat_model(train_data, y_train):
    config = Config()
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


def initialize_gcn_model(train_data, num_classes, batch_mode=False):
    config = Config()

    first_dataset = train_data.dataset[0]
    num_of_features = (
        first_dataset.x.shape[1] * 2 + first_dataset.edge_attr.shape[1]
        if batch_mode else train_data.num_features
    )

    model = GCN(
        optimizer=config.optimizer,
        num_features=num_of_features,
        num_classes=num_classes,
        weight_decay=config.weight_decay,
        dropout=config.dropout,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        lr=config.lr,
        patience=config.patience
    )
    return model


if __name__ == "__main__":
    if not os.path.exists("./results"):
        os.makedirs("./results")

    #print("GAT MODEL")
    #gat_model = initialize_gat_model(train_data, y_train)
    #gat_model.train_model(train_data, test_data)
    #print("=================")

    print("GCN MODEL")
    train_data, test_data = split_data_for_tdg()
    gcn_model = initialize_gcn_model(train_data, num_classes=2, batch_mode=True)
    gcn_model.train_model(train_data, test_data, batch_mode=True)
    print("=================")
