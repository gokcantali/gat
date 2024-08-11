import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader, RandomNodeLoader

from gat.converter import convert_to_graph, create_tdg_graphs_using_window, create_randomly_partitioned_tdg_graphs, \
    create_randomly_partitioned_knn_graphs
from gat.load_data import load_data, save_graph_data, load_graph_data
from gat.model import GAT
from gat.gcn import GCN
from gat.preprocesser import preprocess_df, preprocess_X, preprocess_y, construct_port_scan_label

TEST_SIZE = 0.25
RANDOM_STATE = 42


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


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
    train_data = create_randomly_partitioned_knn_graphs(df_train)
    test_data = create_randomly_partitioned_knn_graphs(df_test)
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


def initialize_gcn_model(num_classes, num_of_features=25):
    config = Config()

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
    device = get_device()

    if not os.path.exists("./results"):
        os.makedirs("./results")

    #print("GAT MODEL")
    #gat_model = initialize_gat_model(train_data, y_train)
    #gat_model.train_model(train_data, test_data)
    #print("=================")

    # print("GCN MODEL - TDG")
    # train_data, test_data = split_data_for_tdg()
    # gcn_model = initialize_gcn_model(
    #     num_classes=2, num_of_features=train_data.num_features
    # )
    # gcn_model.train_model(train_data, test_data, batch_mode=True)
    # print("=================")

    # print("GCN MODEL - kNN")
    # train_data, test_data, _ = split_data()

    # print("===FULL MODE without GPU===")
    # gcn_model = initialize_gcn_model(
    #     num_classes=2, train_data.num_features
    # )
    # gcn_model.train_model(train_data, test_data)
    # print("==============")
    # print("==============")
    # print("==============")
    # print("===FULL MODE with GPU===")
    # train_data.to(device)
    # test_data.to(device)
    # gcn_model = initialize_gcn_model(
    #     num_classes=2, train_data.num_features
    # )
    # gcn_model.to(device)
    # gcn_model.train_model(train_data, test_data)
    # print("==============")

    # print("===BATCH MODE without GPU===")
    # train_loader = DataLoader([train_data])
    # test_loader = DataLoader([test_data])
    # gcn_model = initialize_gcn_model(
    #     num_classes=2, num_of_features=train_data.dataset[0].x.shape[1]
    # )
    # gcn_model.train_model(train_loader, test_loader, batch_mode=True)
    # print("=================")
    # print("=================")
    # print("=================")
    # print("===BATCH MODE with GPU===")
    # train_data.to(device)
    # test_data.to(device)
    # train_loader = DataLoader([train_data])
    # test_loader = DataLoader([test_data])
    # gcn_model = initialize_gcn_model(
    #     num_classes=2, num_of_features=train_data.dataset[0].x.shape[1]
    # )
    # gcn_model.to(device)
    # gcn_model.train_model(train_loader, test_loader, batch_mode=True)
    # print("=================")
    # print("=================")
    # print("=================")

    # df = load_data(sampling_ratio=0.75)
    # df = construct_port_scan_label(df, use_diversity_index=True)
    # df["is_anomaly"] = df["is_anomaly"].replace({"True": 1, "False": 0}).astype(int)
    # graph = convert_to_graph(
    #     X=preprocess_X(df, use_diversity_index=True),
    #     y=df['is_anomaly'].replace({"True": 1, "False": 0}).astype(int)
    # )
    # save_graph_data(graph, 'traces-3ddos-2zap-1scan.75percent.pt')

    graph_data = load_graph_data('traces-3ddos-2zap-1scan.100percent.pt')
    num_parts = 1000

    batches = RandomNodeLoader(graph_data, num_parts=num_parts, shuffle=True)
    train_data, test_data = [], []
    for ind, batch in enumerate(batches):
        if ind < TEST_SIZE * num_parts:
            train_data.append(batch)
        else:
            test_data.append(batch)

    gcn_model = initialize_gcn_model(2, 25)
    gcn_model.train_model(train_data, test_data, batch_mode=True)
