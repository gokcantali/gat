import time
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import mean
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score, balanced_accuracy_score, accuracy_score
)
from torch_geometric.nn import GCNConv

from gat.analysis import plot_metrics, print_epoch_stats
from gat.data_models import Metrics


def calculate_metrics(y_true, y_pred):
    def _safe_division(n, d):
        # handles zero-by-division
        return n / d if d else 0.0

    tp, tn, fp, fn = 0, 0, 0, 0

    for ind in range(len(y_true)):
        truth = y_true[ind]
        pred = y_pred[ind]
        if (truth, pred) == (1, 1):
            tp += 1
        elif (truth, pred) == (0, 1):
            fp += 1
        elif (truth, pred) == (1, 0):
            fn += 1
        elif (truth, pred) == (0, 0):
            tn += 1
        else:
            pass

    return {
        "accuracy": _safe_division((tp + tn), (tp + tn + fp + fn)),
        "precision": _safe_division(tp, (tp + fp)),
        "recall": _safe_division(tp, (tp + fn)),
        "f1_score": _safe_division(2 * tp, (2 * tp + fp + fn)),
    }

class GCN(torch.nn.Module):
    def __init__(
        self, optimizer, num_features, num_classes, weight_decay=1e-3, dropout=0.7,
        hidden_dim=16, epochs=30, lr=0.005, patience=3
    ):
        super(GCN, self).__init__()
        self.epochs = epochs
        self.patience = patience
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.optimizer = optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=self.patience
        )

    def forward(self, x, edge_index):
        #x = self.dropout(x)

        x = F.relu(self.conv1(x, edge_index))
        #x = self.bn1(x)
        #x = self.dropout(x)

        x = F.relu(self.conv2(x, edge_index))
        #x = self.bn3(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=-1)

    def train_epoch(self, data):
        self.train()
        self.optimizer.zero_grad()
        out = self(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        self.optimizer.step()

        _, preds = out.max(dim=1)
        correct = preds.eq(data.y).sum().item()
        accuracy = correct / data.y.size(0)

        precision = precision_score(data.y.cpu(), preds.cpu(), average="weighted")
        recall = recall_score(data.y.cpu(), preds.cpu(), average="weighted")
        f1 = f1_score(data.y.cpu(), preds.cpu(), average="weighted")

        return loss.item(), accuracy, precision, recall, f1

    def train_epoch_batch_mode(self, data_loader, epochs=1):
        self.train()

        num_of_batches = len(data_loader)
        total_loss = 0.0
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        losses = []
        predictions = []
        labels = []
        start_time = time.time()
        for data in data_loader:
            self.optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = F.nll_loss(out, data.y)
            losses.append(loss.detach().item())

            loss.backward()
            self.optimizer.step()

            _, preds = out.max(dim=1)
            predictions += preds.cpu()
            labels += data.y.cpu()

            total_accuracy += balanced_accuracy_score(data.y.cpu(), preds.cpu())
            total_precision += precision_score(data.y.cpu(), preds.cpu(), average="weighted", zero_division=1)
            total_recall += recall_score(data.y.cpu(), preds.cpu(), average="weighted", zero_division=1)
            total_f1 += f1_score(data.y.cpu(), preds.cpu(), average="weighted", zero_division=1)

        end_time = time.time()
        print(f"training time: {(end_time-start_time)}")

        return (
            total_loss / num_of_batches,
            total_accuracy / num_of_batches,
            total_precision / num_of_batches,
            total_recall / num_of_batches,
            total_f1 / num_of_batches
        )

    def validate_epoch(self, data):
        self.eval()
        with torch.no_grad():
            val_out = self(data.x, data.edge_index)
            val_loss = F.nll_loss(val_out, data.y).item()
            _, val_preds = val_out.max(dim=1)
            val_correct = val_preds.eq(data.y).sum().item()
            val_accuracy = val_correct / data.y.size(0)

            val_precision = precision_score(data.y.cpu(), val_preds.cpu(), average="weighted")
            val_recall = recall_score(data.y.cpu(), val_preds.cpu(), average="weighted")
            val_f1 = f1_score(data.y.cpu(), val_preds.cpu(), average="weighted")
            cm = confusion_matrix(data.y.cpu(), val_preds.cpu())

        return val_loss, val_accuracy, val_precision, val_recall, val_f1, cm

    def validate_epoch_batch_model(self, data_loader):
        self.eval()

        num_of_batches = len(data_loader)
        total_loss = 0
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        data_y = []
        pred_y = []
        start_time = time.time()
        for data in data_loader:
            with torch.no_grad():
                val_out = self(data.x, data.edge_index)
                labels = data.y

                total_loss += F.nll_loss(val_out, labels).item()

                _, val_preds = val_out.max(dim=1)
                val_correct = val_preds.eq(labels).sum().item()

                total_accuracy += val_correct / labels.size(0)
                total_precision += precision_score(labels.cpu(), val_preds.cpu(), average="weighted", zero_division=1)
                total_recall += recall_score(labels.cpu(), val_preds.cpu(), average="weighted", zero_division=1)
                total_f1 += f1_score(labels.cpu(), val_preds.cpu(), average="weighted", zero_division=1)
                data_y += labels.cpu()
                pred_y += val_preds.cpu()

        cm = confusion_matrix(data_y, pred_y)
        #tn, fp, fn, tp = cm.ravel()
        #print(f"tp: {tp} - tn: {tn}, fp: {fp}, fn: {fn}")
        end_time = time.time()
        print(f"validation time: {(end_time-start_time)}")

        return (
            total_loss / num_of_batches,
            total_accuracy / num_of_batches,
            total_precision / num_of_batches,
            total_recall / num_of_batches,
            total_f1 / num_of_batches,
            cm
        )

    def train_model(self, train_data, val_data, batch_mode=False, epochs=0):
        metrics = Metrics()
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs or self.epochs):
            start_time = time.time()
            if batch_mode:
                train_loss, train_accuracy, train_precision, train_recall, train_f1 = (
                    self.train_epoch_batch_mode(train_data)
                )
            else:
                train_loss, train_accuracy, train_precision, train_recall, train_f1 = (
                    self.train_epoch(train_data)
                )

            if batch_mode:
                val_loss, val_accuracy, val_precision, val_recall, val_f1, cm = (
                    self.validate_epoch_batch_model(val_data)
                )
            else:
                val_loss, val_accuracy, val_precision, val_recall, val_f1, cm = (
                    self.validate_epoch(val_data)
                )

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
            #print_epoch_stats(
            #    epoch + 1, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy,
            #    train_precision, train_recall, train_f1, val_precision, val_recall, val_f1, cm)

            # Adjust learning rate based on validation loss
            self.scheduler.step(val_loss)
            print(f'Epoch {epoch + 1}, Current Learning Rate: {self.scheduler.get_last_lr()}')

            # Early stopping check to prevent overfitting
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping")
                    break

        self.load_state_dict(torch.load("best_model.pth"))
        #plot_metrics(metrics)
        return metrics

    def test_model(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        _, pred = out.max(dim=1)
        return pred

    def test_model_batch_mode(self, data):
        predictions = []
        losses = []
        labels = []
        for batch in data:
            labels += batch.y

            out = self(batch.x, batch.edge_index)
            loss = F.nll_loss(out, batch.y).item()
            losses.append(loss)
            _, pred = out.max(dim=1)
            predictions += pred

        perf_metrics = {
            "accuracy": accuracy_score(predictions, labels),
            "precision": precision_score(labels, predictions, average="weighted", zero_division=1),
            "recall": recall_score(labels, predictions, average="weighted", zero_division=1),
            "f1_score": f1_score(labels, predictions, average="weighted", zero_division=1),
        }
        return predictions, mean(losses), perf_metrics

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)
        torch.save(state_dict, "best_model.pth")

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
