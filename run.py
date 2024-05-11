import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

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

def plot_loss_and_accuracy(loss_values, accuracy_values):
    _, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(range(1, EPOCHS+1), loss_values, marker='o', linestyle='-', color='b')
    ax2.plot(range(1, EPOCHS+1), accuracy_values, marker='o', linestyle='-', color='r')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax2.set_ylabel('Accuracy', color='r')
    plt.title('Loss and Accuracy per Epoch')
    plt.grid(True)
    plt.savefig('./results/loss_and_accuracy_per_epoch.png')
    plt.close()

def train_model(model, train_data):
    loss_values = []
    accuracy_values = []
    for epoch in range(EPOCHS):
        loss, accuracy = model.train_model(train_data)
        loss_values.append(loss)
        accuracy_values.append(accuracy)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
    plot_loss_and_accuracy(loss_values, accuracy_values)

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
train_model(model, train_data)
metrics = evaluate_model(model, test_data)
print_metrics(metrics)
