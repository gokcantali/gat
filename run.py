import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader, RandomNodeLoader

from gat.converter import convert_to_graph, create_tdg_graphs_using_window, create_randomly_partitioned_tdg_graphs, \
    create_randomly_partitioned_knn_graphs
from gat.load_data import load_data, save_graph_data, load_graph_data
from gat.model import GAT
from gat.gcn import GCN
from gat.preprocesser import preprocess_df, preprocess_X, preprocess_y, construct_port_scan_label

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
TRAIN_SIZE = 1 - VALIDATION_SIZE - TEST_SIZE


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class Config:
    optimizer = torch.optim.AdamW
    lr = 0.03524
    weight_decay = 0.00048463407384332236
    epochs = 29
    patience = 7
    hidden_dim = 38
    dropout = 0.4416

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


def initialize_gcn_model(num_classes, num_of_features=25, config=None):
    if config is None:
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


def create_stratified_knn_graphs(file_name):
    df = load_data(sampling_ratio=1, file_path=Path(f"data/{file_name}.csv"))
    df = construct_port_scan_label(df, use_diversity_index=True)
    df["is_anomaly"] = df["is_anomaly"].replace({"True": 1, "False": 0}).astype(int)

    # preprocess the data file and creates dataframe
    X = preprocess_X(df, use_diversity_index=True)
    y = df['anomaly_class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE
    )

    # creates the graph for the training phase
    train_graph = convert_to_graph(
        X=X_train,
        y=y_train
    )
    save_graph_data(train_graph, f'{file_name}-train.pt')

    # creates the graph for the testing phase
    test_graph = convert_to_graph(
        X=X_test,
        y=y_test
    )
    save_graph_data(test_graph, f'{file_name}-test.pt')


def run_with_different_training_and_test_graphs(
    train_graph, test_graph, use_pretrained_model=False
):
    num_parts = 50

    test_graph_data = load_graph_data(test_graph)
    test_graph_data.x[:, 18] = torch.zeros_like(test_graph_data.x[:, 18])
    test_graph_data.x[:, 19] = torch.zeros_like(test_graph_data.x[:, 19])

    test_data = []
    test_batches = RandomNodeLoader(test_graph_data, num_parts=num_parts, shuffle=True)
    y_true = []
    for _, batch in enumerate(test_batches):
        test_data.append(batch)
        y_true += batch.y

    gcn_model = initialize_gcn_model(4, 25)
    if use_pretrained_model is True:
        gcn_model.load_state_dict(torch.load('best_model_FL.pt'))

    else:
        train_graph_data = load_graph_data(train_graph)
        train_graph_data.x[:, 18] = torch.zeros_like(train_graph_data.x[:, 18])
        train_graph_data.x[:, 19] = torch.zeros_like(train_graph_data.x[:, 19])

        start_time = time.time()
        train_data, validation_data = [], []
        train_batches = RandomNodeLoader(train_graph_data, num_parts=num_parts, shuffle=True)
        for ind, batch in enumerate(train_batches):
            if ind < (VALIDATION_SIZE / (VALIDATION_SIZE + TRAIN_SIZE)) * num_parts:
                validation_data.append(batch)
            else:
                train_data.append(batch)
        end_time = time.time()
        print(f"Time for batching data: {end_time - start_time}")

        start_time = time.time()
        training_metrics = gcn_model.train_model(train_data, validation_data, batch_mode=True)
        end_time = time.time()
        print(f"Total Training Time: {end_time - start_time}")
        print("=======TRAINING COMPLETED!=======\n")

    start_time = time.time()
    y_pred, _, _ = gcn_model.test_model_batch_mode(test_data)
    end_time = time.time()
    print(f"Total Testing Time: {end_time - start_time}")
    print("=====TEST RESULTS=======")
    print(confusion_matrix(y_true, y_pred))
    # return confusion_matrix(y_true, y_pred)

    experiment_result_file_name = f"train-{train_graph.split('-')[0]}"
    experiment_result_file_name += f"-test-{test_graph.split('-')[0]}"
    if use_pretrained_model is True:
        experiment_result_file_name += "-FL"
    experiment_result_file_name += ".txt"

    with open(experiment_result_file_name, 'a') as file:
        file.write(str(confusion_matrix(y_true, y_pred)) + '\n')


def run(config=None, mode='train'):
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

    for _ in range(10):
        graph_data = load_graph_data('worker1-traces-75min.pt')
        graph_data.x[:, 18] = torch.zeros_like(graph_data.x[:, 18])
        graph_data.x[:, 19] = torch.zeros_like(graph_data.x[:, 19])
        num_parts = 1000

        start_time = time.time()
        batches = RandomNodeLoader(graph_data, num_parts=num_parts, shuffle=True)
        train_data, validation_data, test_data = [], [], []
        y_true = []
        for ind, batch in enumerate(batches):
            if ind < TEST_SIZE * num_parts:
                test_data.append(batch)
                y_true += batch.y
            elif ind < (TEST_SIZE + VALIDATION_SIZE) * num_parts:
                validation_data.append(batch)
            else:
                train_data.append(batch)
        end_time = time.time()
        print(f"Time for batching data: {end_time - start_time}")

        start_time = time.time()
        gcn_model = initialize_gcn_model(4, 25)
        gcn_model.load_state_dict(torch.load('best_model.pth'))
        training_metrics = gcn_model.train_model(train_data, validation_data, batch_mode=True)
        end_time = time.time()
        print(f"Total Training Time: {end_time - start_time}")
        print("=======TRAINING COMPLETED!=======\n")
        # if mode == 'train':
        #     return training_metrics

        start_time = time.time()
        y_pred, _, _ = gcn_model.test_model_batch_mode(test_data)
        end_time = time.time()
        print(f"Total Testing Time: {end_time - start_time}")
        print("=====TEST RESULTS=======")
        print(confusion_matrix(y_true, y_pred))
        # return confusion_matrix(y_true, y_pred)

        with open('worker1-results-fl-params.txt', 'a') as file:
            file.write(str(confusion_matrix(y_true, y_pred)) + '\n')


if __name__ == "__main__":
    # run(mode='test')
    # create_stratified_knn_graphs("master-traces-75min")
    # run_with_different_training_and_test_graphs(
    #     "worker3-traces-75min.pt", "worker4-traces-75min.pt",
    #     use_pretrained_model=True
    # )

    TRIALS = 10
    TRAIN_WORKER_IND = 0
    for use_pretrained_model in [False, True]:
        for test_worker_ind in range(5):
            for _ in range(TRIALS):
                run_with_different_training_and_test_graphs(
                    train_graph=f"worker{TRAIN_WORKER_IND}-traces-75min.pt",
                    test_graph=f"worker{test_worker_ind}-traces-75min.pt",
                    use_pretrained_model=use_pretrained_model
                )
