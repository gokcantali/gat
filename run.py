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

# Display unique values of labels and their data types for inspection
print(y.unique())
print(y.dtypes)

# Mapping for converting boolean or string labels to integers
mapping = {'False': 0, 'True': 1, False: 0, True: 1}  # Adjust based on data
y_int = y.replace(mapping).astype(int)  # Apply mapping and convert to integer

# Split the dataset into training and testing sets, ensuring each class is
# represented proportionally in both sets using stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_int, test_size=0.25, stratify=y_int, random_state=42)

# Function to convert DataFrame to graph format suitable for GAT
def convert_to_graph(X, y):
    # Converts feature data to numeric type, handling non-numeric entries
    X_cleaned = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_tensor = torch.tensor(X_cleaned.values, dtype=torch.float)  # To tensor
    y_tensor = torch.tensor(y.values, dtype=torch.long)  # Labels to tensor
    num_nodes = len(X_tensor)  # Total number of nodes
    
    # Initialize masks for training, testing, validation
    masks = [torch.zeros(num_nodes, dtype=torch.bool) for _ in range(3)]
    limits = [int(0.8 * num_nodes), int(0.9 * num_nodes), num_nodes]
    for mask, limit in zip(masks, limits):
        mask[:limit] = True
        np.random.shuffle(mask.numpy())  # Shuffle masks to randomize selection

    # Define edge index for creating graph connections (simple chain for example)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
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
