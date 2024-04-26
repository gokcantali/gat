import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, optimizer, num_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=True, dropout=0.6)
        self.optimizer = optimizer = optimizer(self.parameters(), lr=0.005, weight_decay=5e-4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def train_model(self, data):
        self.train()  # Set model to training mode
        self.optimizer.zero_grad()  # Clear previous gradients
        out = self(data)  # Forward pass
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # Compute loss
        loss.backward()  # Backpropagate the loss
        self.optimizer.step()  # Update model weights
        return loss.item()

    def test_model(self, data):
        self.eval()  # Set model to evaluation mode
        out = self(data)  # Forward pass
        _, pred = out.max(dim=1)  # Get predictions
        return pred  # Return accuracy and predictions
