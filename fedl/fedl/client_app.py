from pathlib import Path

import os
import torch
from flwr.client import NumPyClient, Client, ClientApp
from flwr.client.mod import secaggplus_mod
from flwr.common import Context
from flwr.client.mod.localdp_mod import LocalDpMod
from torch import load
from torch_geometric.loader import RandomNodeLoader

from run import initialize_gcn_model


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, testloader, **kwargs):
        super().__init__(**kwargs)
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
        _, loss, accuracy = self.net.test_model_batch_mode(self.testloader)
        return loss, len(self.testloader), {"accuracy": float(accuracy)}


def construct_flower_client(client_id, context):
    # Load model
    net = initialize_gcn_model(num_classes=4)

    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node

    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    graph_data = load(Path(f'{root}/data/graph/multi-class-traces-3ddos-2zap-1scan.pt'))
    graph_data.x[:, 18] = torch.zeros_like(graph_data.x[:, 18])
    graph_data.x[:, 19] = torch.zeros_like(graph_data.x[:, 19])
    num_parts = 1000

    batches = RandomNodeLoader(graph_data, num_parts=num_parts, shuffle=True)
    local_part_start, local_part_end = num_parts * client_id // 5, num_parts * (client_id + 1) // 5
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
    return FlowerClient(
        net, train_loader, validation_loader, test_loader,
    ).to_client()


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    partition_id = context.node_config["partition-id"]

    # Construct the client
    flower_client = construct_flower_client(
        client_id=partition_id, context=context
    )
    return flower_client


# Create an instance of the mod with the required params
local_dp_obj = LocalDpMod(
    0.8, 0.2, 0.0001, 0.0001
)

# Create the ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,  # Comment-out to disable SecAgg+
        local_dp_obj  # Comment-out to disable DP
    ],
)
