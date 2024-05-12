import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from gat.converter import convert_to_graph
from gat.model import GAT
from gat.preprocesser import preprocess_df, preprocess_X, preprocess_y

TEST_SIZE = 0.25
RANDOM_STATE = 42

class Config:
    optimizer = torch.optim.AdamW
    lr = 0.04
    weight_decay = 0.00048
    epochs = 30
    patience = 5
    hidden_dim = 32
    dropout = 0.4

def split_data():
    df = preprocess_df()
    X = preprocess_X(df)
    y = preprocess_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    train_data = convert_to_graph(X_train, y_train)
    test_data = convert_to_graph(X_test, y_test)
    return train_data, test_data, y_train

def initialize_model(train_data, y_train):
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

if not os.path.exists("./results"):
    os.makedirs("./results")
train_data, test_data, y_train = split_data()
model = initialize_model(train_data, y_train)
model.train_model(train_data, test_data)
