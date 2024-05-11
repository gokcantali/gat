import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from gat.analysis import plot_metrics
from gat.converter import convert_to_graph
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

def train_model(model, train_data, val_data, epochs):
    train_loss_values = []
    val_loss_values = []
    train_accuracy_values = []
    val_accuracy_values = []
    train_precision_values = []
    train_recall_values = []
    train_f1_values = []
    val_precision_values = []
    val_recall_values = []
    val_f1_values = []
    epoch_values = []

    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = model.train_model(train_data)
        train_precision_values.append(train_precision)
        train_recall_values.append(train_recall)
        train_f1_values.append(train_f1)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_out = model(val_data)
            val_loss = F.nll_loss(val_out, val_data.y).item()
            _, val_preds = val_out.max(dim=1)
            val_correct = val_preds.eq(val_data.y).sum().item()
            val_accuracy = val_correct / val_data.y.size(0)

            # Compute precision, recall, F1-score and confusion matrix for validation
            val_precision = precision_score(val_data.y.cpu(), val_preds.cpu(), average='weighted')
            val_recall = recall_score(val_data.y.cpu(), val_preds.cpu(), average='weighted')
            val_f1 = f1_score(val_data.y.cpu(), val_preds.cpu(), average='weighted')
            cm = confusion_matrix(val_data.y.cpu(), val_preds.cpu())
            val_precision_values.append(val_precision)
            val_recall_values.append(val_recall)
            val_f1_values.append(val_f1)

        epoch_values.append(epoch + 1)
        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)
        train_accuracy_values.append(train_accuracy)
        val_accuracy_values.append(val_accuracy)

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}, Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.4f}')
        print(f'Train Precision: {train_precision:.4f}')
        print(f'Train Recall: {train_recall:.4f}')
        print(f'Train F1-Score: {train_f1:.4f}')
        print(f'Validation Precision: {val_precision:.4f}')
        print(f'Validation Recall: {val_recall:.4f}')
        print(f'Validation F1-Score: {val_f1:.4f}')
        print(f'Confusion Matrix:\n{cm}')
        print('--------------------------------------------------')

    plot_metrics(
        epoch_values, train_loss_values, val_loss_values, train_accuracy_values,
        val_accuracy_values, train_precision_values, train_recall_values, train_f1_values,
        val_precision_values, val_recall_values, val_f1_values)

def evaluate_model(model, test_data):
    pred = model.test_model(test_data)
    all_labels = np.unique(test_data.y.numpy())
    metrics = {
        'conf_matrix': confusion_matrix(
            test_data.y.numpy(), pred.numpy(), labels=all_labels),
        'precision': precision_score(
            test_data.y.numpy(), pred.numpy(), average='weighted', labels=all_labels),
        'recall': recall_score(
            test_data.y.numpy(), pred.numpy(), average='weighted', labels=all_labels),
        'f1': f1_score(
            test_data.y.numpy(), pred.numpy(), average='weighted', labels=all_labels),
        'test_accuracy': accuracy_score(
            test_data.y.numpy(), pred.numpy())
    }
    return metrics

def print_metrics(metrics):
    print("Confusion Matrix:\n", metrics['conf_matrix'])
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

train_data, test_data, y_train = split_data()
model = initialize_model(train_data, y_train)
train_model(model, train_data, test_data, EPOCHS)
# metrics = evaluate_model(model, test_data)
# print_metrics(metrics)
