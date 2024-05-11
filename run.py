import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from gat.analysis import plot_metrics
from gat.converter import convert_to_graph
from gat.data_models import Metrics
from gat.model import GAT
from gat.preprocesser import preprocess_df, preprocess_X, preprocess_y

EPOCHS = 30
TEST_SIZE = 0.25
RANDOM_STATE = 42

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
    model = GAT(
        torch.optim.Adam,
        num_features=train_data.num_features,
        num_classes=len(np.unique(y_train))
    )
    return model

def train_epoch(model, data):
    model.train()
    train_loss, train_accuracy, train_precision, train_recall, train_f1 = model.train_model(data)
    return train_loss, train_accuracy, train_precision, train_recall, train_f1

def validate_epoch(model, data):
    model.eval()
    with torch.no_grad():
        val_out = model(data)
        val_loss = F.nll_loss(val_out, data.y).item()
        _, val_preds = val_out.max(dim=1)
        val_correct = val_preds.eq(data.y).sum().item()
        val_accuracy = val_correct / data.y.size(0)

        val_precision = precision_score(data.y.cpu(), val_preds.cpu(), average='weighted')
        val_recall = recall_score(data.y.cpu(), val_preds.cpu(), average='weighted')
        val_f1 = f1_score(data.y.cpu(), val_preds.cpu(), average='weighted')
        cm = confusion_matrix(data.y.cpu(), val_preds.cpu())

    return val_loss, val_accuracy, val_precision, val_recall, val_f1, cm

def print_epoch_stats(epoch, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy,
                      train_precision, train_recall, train_f1, val_precision, val_recall, val_f1, cm):
    print(f'Epoch {epoch}, Time: {epoch_time:.2f}s')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    print(f'Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1-Score: {train_f1:.4f}')
    print(f'Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Validation F1-Score: {val_f1:.4f}')
    print(f'Confusion Matrix:\n{cm}')
    print('--------------------------------------------------')

def train_model(model, train_data, val_data, epochs):
    metrics = Metrics()

    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train_epoch(model, train_data)
        val_loss, val_accuracy, val_precision, val_recall, val_f1, cm = validate_epoch(model, val_data)

        metrics.train_loss.append(train_loss)
        metrics.val_loss.append(val_loss)
        metrics.train_accuracy.append(train_accuracy)
        metrics.val_accuracy.append(val_accuracy)
        metrics.train_precision.append(train_precision)
        metrics.train_recall.append(train_recall)
        metrics.train_f1.append(train_f1)
        metrics.val_precision.append(val_precision)
        metrics.val_recall.append(val_recall)
        metrics.val_f1.append(val_f1)
        metrics.epoch_values.append(epoch + 1)

        epoch_time = time.time() - start_time
        print_epoch_stats(epoch + 1, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy,
                          train_precision, train_recall, train_f1, val_precision, val_recall, val_f1, cm)

    plot_metrics(metrics)

train_data, test_data, y_train = split_data()
model = initialize_model(train_data, y_train)
train_model(model, train_data, test_data, EPOCHS)
