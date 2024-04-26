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
from gat.encoder import boolean_string_to_int
from gat.load_data import load_data
from gat.model import GAT

EPOCHS = 20

X, y, header = load_data()
y = boolean_string_to_int(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
train_data = convert_to_graph(X_train, y_train)
test_data = convert_to_graph(X_test, y_test)
model = GAT(torch.optim.Adam, num_features=train_data.num_features, num_classes=len(np.unique(y_train)))

for epoch in range(EPOCHS):
    loss = model.train_model(train_data)
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

accuracy, pred = model.test_model(test_data)
all_labels = np.unique(y)
conf_matrix = confusion_matrix(test_data.y.numpy(), pred.numpy(), labels=all_labels)
precision = precision_score(test_data.y.numpy(), pred.numpy(), average='weighted', labels=all_labels)
recall = recall_score(test_data.y.numpy(), pred.numpy(), average='weighted', labels=all_labels)
f1 = f1_score(test_data.y.numpy(), pred.numpy(), average='weighted', labels=all_labels)
test_accuracy = accuracy_score(test_data.y.numpy(), pred.numpy())

print("Confusion Matrix:\n", conf_matrix)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
