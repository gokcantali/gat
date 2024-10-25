from pathlib import Path

import torch
from flwr.client import NumPyClient, Client, ClientApp
from flwr.common import Context
from sklearn.metrics import accuracy_score
from torch import load
from torch_geometric.loader import RandomNodeLoader

from gat.load_data import load_graph_data
from run import initialize_gcn_model


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

    def get_parameters(self, config):
        return self.net.get_parameters()

    def fit(self, parameters, config):
        self.net.set_parameters(parameters)
        self.net.train_model(self.trainloader, self.valloader, batch_mode=True, epochs=1)
        return self.net.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters)
        #preds = self.net.test_model_batch_mode(self.testloader)
        #accuracy = accuracy_score(preds, [test.y for test in self.testloader])
        #return float(1-accuracy), len(self.testloader), {"accuracy": float(accuracy)}
        # Calculate accuracy and loss
        return 0, len(self.testloader), {"accuracy": 0.0}


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = initialize_gcn_model(num_classes=4)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]

    graph_data = load(Path('../data/graph/multi-class-traces-3ddos-2zap-1scan.pt'))
    graph_data.x[:, 18] = torch.zeros_like(graph_data.x[:, 18])
    graph_data.x[:, 19] = torch.zeros_like(graph_data.x[:, 19])
    num_parts = 1000

    batches = RandomNodeLoader(graph_data, num_parts=num_parts, shuffle=True)
    local_part_start, local_part_end = num_parts * partition_id // 5, num_parts * (partition_id + 1) // 5
    train_loader, validation_loader, test_loader = [], [], []
    y_true = []
    for ind, batch in enumerate(batches):
        if ind < local_part_start or ind >= local_part_end:
            continue

        if ind < 0.8 * (local_part_end - local_part_start) + local_part_start:
            train_loader.append(batch)
            y_true += batch.y
        elif ind < 0.9 * (local_part_end - local_part_start) + local_part_start:
            validation_loader.append(batch)
        else:
            test_loader.append(batch)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, train_loader, validation_loader, test_loader).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)
