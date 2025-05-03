import numpy as np
import torch
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from gat.load_data import load_data
from gat.preprocesser import preprocess_df, preprocess_X, preprocess_y, process_timestamps, create_sequences, \
    construct_port_scan_label


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, hyperparams=None):
        super(RNNClassifier, self).__init__()
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

        if hyperparams and type(hyperparams) is dict:
            self.batch_size = hyperparams.get('batch_size', 32)
            self.learning_rate = hyperparams.get('learning_rate', 0.001)
            self.num_epochs = hyperparams.get('num_epochs', 10)
            self.patience = hyperparams.get('patience', 3)

    def forward(self, x):
        out, _ = self.rnn(x)  # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]   # Take last time step's output
        out = self.fc(out)
        return out

    def train_rnn(self, X_train, y_train, X_val, y_val):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.batch_size
        )
        print("Start training")

        # Training loop
        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0
            correct = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()

            acc = correct / len(train_dataset)
            print(f"Epoch {epoch + 1} | Loss: {total_loss:.4f} | Accuracy: {acc:.4f}")


class NetworkTrafficRNN(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers,
        num_classes, dropout, hyperparams=None
    ):
        super(NetworkTrafficRNN, self).__init__()
        if hyperparams is None:
            hyperparams = {}
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN layer
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_size, num_classes)

        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)

        # Computation Device
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set hyperparameters
        if hyperparams and type(hyperparams) is dict:
            self.batch_size = hyperparams.get('batch_size', 32)
            self.learning_rate = hyperparams.get('learning_rate', 0.001)
            self.num_epochs = hyperparams.get('num_epochs', 10)
            self.patience = hyperparams.get('patience', 3)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(self.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

    def train_rnn(self, X_train, y_train, X_val, y_val):
        # Create DataLoader
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=self.batch_size
        )
        val_data = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            val_data, shuffle=True, batch_size=self.batch_size
        )

        #class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.numpy())
        #class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        # Loss and optimizer
        #criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Loss and optimizer
        # criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Device configuration
        # self.to(self.device)

        # # Training loop
        # for epoch in range(self.num_epochs):
        #     self.train()
        #     for batch_idx, (data, targets) in enumerate(train_loader):
        #         data = data.to(self.device)
        #         targets = targets.to(self.device)
        #
        #         # Forward pass
        #         scores = self(data)
        #         loss = criterion(scores, targets)
        #
        #         # Backward pass
        #         optimizer.zero_grad()
        #         loss.backward()
        #
        #         # Gradient descent
        #         optimizer.step()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                # inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    #inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Calculate metrics
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total

            print(f'Epoch {epoch + 1}/{self.num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc:.4f}')

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), '../rnn_best_model.pth')  # Save best model
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered!")
                    self.load_state_dict(torch.load("../rnn_best_model.pth"))
                    break

    def evaluate_rnn(self, X, y):
        self.eval()
        with torch.no_grad():
            #X = X.to(self.device)
            #y = y.to(self.device)

            outputs = self(X)
            _, predicted = torch.max(outputs.data, 1)

            correct = (predicted == y).sum().item()
            total = y.size(0)
            accuracy = correct / total

            print(f'Test Accuracy: {accuracy * 100:.2f}%')

            # You can add more metrics like precision, recall, F1-score
            from sklearn.metrics import classification_report
            print(classification_report(y.cpu(), predicted.cpu()))


if __name__ == '__main__':
    df = load_data(
        "../data/subsample/traces-benign0.20-dos0.71-port0.92-zap1.00.csv",
    )
    df = df.sort_values(by=['index'])
    df = construct_port_scan_label(df, use_diversity_index=False)
    df["is_anomaly"] = df["is_anomaly"].replace({"True": 1, "False": 0}).astype(int)

    data = preprocess_X(
        df, use_diversity_index=False, keep_timestamp=True
    )
    data = process_timestamps(data)
    data['label'] = preprocess_y(df)

    sequence_length = 10
    X, y = create_sequences(
        data, sequence_length
    )

    # Normalization is a must here!
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True)
    X_norm = (X - mean) / (std + 1e-6)

    print("sequences")
    print(X)

    # Construct train/validation/test splits
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_norm, y, test_size=0.2, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2/0.8, shuffle=True
    )

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)

    means = X_train.mean(dim=(0, 1), keepdim=True)
    stds = X_train.std(dim=(0, 1), keepdim=True)
    X_train = (X_train - means) / (stds + 1e-6)

    # Configure the model layers and hyperparams
    input_size = X_train.shape[2]  # Number of features
    hidden_size = 128
    num_layers = 2
    num_classes = 4
    dropout = 0.05

    hyperparams = {
        "learning_rate": 0.01,
        "batch_size": 512,
        "num_epochs": 150,
        "patience": 10
    }

    model = NetworkTrafficRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        hyperparams=hyperparams
    )
    # model = RNNClassifier(
    #     input_size=input_size,
    #     hidden_size=hidden_size,
    #     num_classes=num_classes,
    #     hyperparams=hyperparams
    # )

    # Train the model
    model.train_rnn(
        X_train, y_train, X_val, y_val
    )

    # Evaluate the model
    model.evaluate_rnn(X_test, y_test)

    # Save the model
    torch.save(model.state_dict(), '../rnn_model.pth')
