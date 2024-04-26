# Importing necessary libraries and modules for the project
from gat.load_data import load_data  # Custom function to load data
from sklearn.model_selection import train_test_split  # Splitting data
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from torch_geometric.data import Data  # PyTorch Geometric data handling
import torch
import torch.nn.functional as F  # PyTorch's functional interface
from gat.model import GAT  # Graph Attention Network model from the gat package
import pandas as pd
import numpy as np

# Load data using a custom function from the gat package
X, y, header = load_data()

# Mapping for converting y string values to integers
mapping = {'False': 0, 'True': 1}
# Apply the mapping and directly convert to integer
y_int = y.map(mapping).astype(int)

# Now you can proceed to split the data
# stratify makes sure that the distribution of classes is similar in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_int, test_size=0.25, stratify=y_int, random_state=42)

# Function to convert DataFrame to graph format suitable for GAT
def convert_to_graph(X, y):
    # Converts feature data to numeric type, handling non-numeric entries
    # This is necessary because neural networks require numerical input for
    # computations, and any non-numeric data would cause errors during training.
    X_cleaned = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Converts the cleaned DataFrame to a tensor, which is the required format
    # for data in PyTorch. This tensor will represent the node features in the graph.
    X_tensor = torch.tensor(X_cleaned.values, dtype=torch.float)

    # Converts the label data to a tensor, ensuring it's in long format for
    # compatibility with classification tasks in PyTorch.
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    # Total number of nodes calculated from the number of rows in the tensor.
    # This is used to initialize the masks that determine which nodes are used for
    # training, validation, and testing.
    num_nodes = len(X_tensor)

    # Initializes masks as boolean tensors for training, testing, and validation.
    # Each mask is of the same length as the number of nodes, initially set to False.
    masks = [torch.zeros(num_nodes, dtype=torch.bool) for _ in range(3)]

    # Defines limits for each mask based on desired proportions:
    # - 80% of data for training,
    # - additional 10% for testing,
    # - remaining 10% for validation.
    limits = [int(0.8 * num_nodes), int(0.9 * num_nodes), num_nodes]

    # Assigns True up to the specified limit for each mask and shuffles it.
    # Shuffling ensures that the selection of nodes for training, testing, and
    # validation is randomized, which helps in reducing bias and overfitting.
    for mask, limit in zip(masks, limits):
        mask[:limit] = True
        np.random.shuffle(mask.numpy())

    # Defines the edge index, which specifies the connections between nodes.
    # Here, a simple bi-directional connection between two consecutive nodes is used.
    # This might need to be customized based on the actual topology of the graph
    # relevant to your specific dataset or domain problem.
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    # Returns a Data object containing:
    # - node features (x),
    # - edge connections (edge_index),
    # - labels (y),
    # - and the three masks specifying which nodes to use for training, testing, and validation.
    return Data(x=X_tensor, edge_index=edge_index, y=y_tensor,
                train_mask=masks[0], test_mask=masks[1], val_mask=masks[2])

# Convert both training and testing data to graph format
train_data = convert_to_graph(X_train, y_train)
test_data = convert_to_graph(X_test, y_test)

# Initialize GAT model specifying the number of features and classes
model = GAT(num_features=train_data.num_features, num_classes=len(np.unique(y_train)))

# Define optimizer with specific learning rate and weight decay for regularization
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Define training function to update model weights
def train(model, data):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear previous gradients
    out = model(data)  # Forward pass
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # Compute loss
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update model weights
    return loss.item()

# Define testing function to evaluate model performance
def test(model, data):
    model.eval()  # Set model to evaluation mode
    out = model(data)  # Forward pass
    _, pred = out.max(dim=1)  # Get predictions
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    acc = correct / int(data.test_mask.sum())  # Calculate accuracy
    return acc, pred  # Return accuracy and predictions

# Train model for a specified number of epochs
for epoch in range(20):
    loss = train(model, train_data)
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

# Test the model and capture predictions
accuracy, pred = test(model, test_data)

# Compute confusion matrix and additional metrics
all_labels = np.unique(y_int)  # Ensure all_labels is an integer array
conf_matrix = confusion_matrix(test_data.y.numpy(), pred.numpy(), labels=all_labels)
precision = precision_score(test_data.y.numpy(), pred.numpy(), average='weighted', labels=all_labels)
recall = recall_score(test_data.y.numpy(), pred.numpy(), average='weighted', labels=all_labels)
f1 = f1_score(test_data.y.numpy(), pred.numpy(), average='weighted', labels=all_labels)
test_accuracy = accuracy_score(test_data.y.numpy(), pred.numpy())

# Print all computed metrics for model evaluation
print("Confusion Matrix:\n", conf_matrix)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
