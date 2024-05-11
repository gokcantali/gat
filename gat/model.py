import time

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.nn import GATConv

from gat.analysis import plot_metrics, print_epoch_stats
from gat.data_models import Metrics


class GAT(torch.nn.Module):
    def __init__(self, optimizer, num_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=True, dropout=0.6)
        self.optimizer = optimizer(self.parameters(), lr=0.005, weight_decay=5e-4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def train_epoch(self, data):
        self.train()
        self.optimizer.zero_grad()
        out = self(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        self.optimizer.step()

        _, preds = out.max(dim=1)
        correct = preds.eq(data.y).sum().item()
        accuracy = correct / data.y.size(0)

        precision = precision_score(data.y.cpu(), preds.cpu(), average='weighted')
        recall = recall_score(data.y.cpu(), preds.cpu(), average='weighted')
        f1 = f1_score(data.y.cpu(), preds.cpu(), average='weighted')

        return loss.item(), accuracy, precision, recall, f1

    def validate_epoch(self, data):
        self.eval()
        with torch.no_grad():
            val_out = self(data)
            val_loss = F.nll_loss(val_out, data.y).item()
            _, val_preds = val_out.max(dim=1)
            val_correct = val_preds.eq(data.y).sum().item()
            val_accuracy = val_correct / data.y.size(0)

            val_precision = precision_score(data.y.cpu(), val_preds.cpu(), average='weighted')
            val_recall = recall_score(data.y.cpu(), val_preds.cpu(), average='weighted')
            val_f1 = f1_score(data.y.cpu(), val_preds.cpu(), average='weighted')
            cm = confusion_matrix(data.y.cpu(), val_preds.cpu())

        return val_loss, val_accuracy, val_precision, val_recall, val_f1, cm

    def train_model(self, train_data, val_data, epochs, patience=10):
        metrics = Metrics()
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()
            train_loss, train_accuracy, train_precision, train_recall, train_f1 = self.train_epoch(
                train_data)
            val_loss, val_accuracy, val_precision, val_recall, val_f1, cm = self.validate_epoch(
                val_data)

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
            print_epoch_stats(
                epoch + 1, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy,
                train_precision, train_recall, train_f1, val_precision, val_recall, val_f1, cm)

            # early stopping check to prevent overfitting
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break

        self.load_state_dict(torch.load('best_model.pth'))
        plot_metrics(metrics)

    def test_model(self, data):
        self.eval()
        out = self(data)
        _, pred = out.max(dim=1)
        return pred
