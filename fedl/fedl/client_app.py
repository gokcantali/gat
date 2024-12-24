import math
from pathlib import Path
# from codecarbon import track_emissions

import os
import torch
from codecarbon import track_emissions, EmissionsTracker
from flwr.client import NumPyClient, Client, ClientApp
from flwr.client.mod import secaggplus_mod
from flwr.common import Context, Config, Scalar
from flwr.client.mod.localdp_mod import LocalDpMod
from torch import load
from torch_geometric.loader import RandomNodeLoader

from run import initialize_gcn_model


TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10
TRAIN_SIZE = 1 - VALIDATION_SIZE - TEST_SIZE


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, testloader, **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        return self.get_context().node_config

    def get_parameters(self, config):
        return self.net.get_parameters()

    # @track_emissions(
    #     # api_endpoint= "http://localhost:8000",
    #     measure_power_secs=10,
    #     # api_call_interval=5,
    #     experiment_id="2ef8bb00-570a-4110-b59f-68f8b5e5fd2a",
    #     save_to_api=False,
    #     allow_multiple_runs=True
    # )
    def fit(self, parameters, config):
        tracker = EmissionsTracker(
            measure_power_secs=10,
            experiment_id="2ef8bb00-570a-4110-b59f-68f8b5e5fd2a",
            save_to_api=False,
            allow_multiple_runs=True
        )
        self.net.set_parameters(parameters)
        tracker.start()
        self.net.train_model(self.trainloader, self.valloader, batch_mode=True, epochs=1)
        emissions = tracker.stop()
        self.emissions = emissions if not math.isnan(emissions) else emissions
        return self.net.get_parameters(), len(self.trainloader), {"carbon": emissions}

    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters)
        _, loss, perf_metrics = self.net.test_model_batch_mode(self.testloader)
        print("METRICS OF CLIENT:")
        print(perf_metrics)
        return loss, len(self.testloader), perf_metrics


def construct_flower_client(client_id, context):
    # Load model
    net = initialize_gcn_model(num_classes=4)

    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    num_parts = 50
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    test_graph_data = load(Path(f'{root}/data/graph/worker{client_id}-traces-75min-test.pt'))
    test_graph_data.x[:, 18] = torch.zeros_like(test_graph_data.x[:, 18])
    test_graph_data.x[:, 19] = torch.zeros_like(test_graph_data.x[:, 19])

    test_loader, y_true = [], []
    test_batches = RandomNodeLoader(test_graph_data, num_parts=num_parts, shuffle=True)
    for _, batch in enumerate(test_batches):
        test_loader.append(batch)
        y_true += batch.y

    train_graph_data = load(Path(f'{root}/data/graph/worker{client_id}-traces-75min-train.pt'))
    train_graph_data.x[:, 18] = torch.zeros_like(train_graph_data.x[:, 18])
    train_graph_data.x[:, 19] = torch.zeros_like(train_graph_data.x[:, 19])

    train_loader, validation_loader = [], []
    train_batches = RandomNodeLoader(train_graph_data, num_parts=num_parts, shuffle=True)
    for ind, batch in enumerate(train_batches):
        if ind < (VALIDATION_SIZE / (VALIDATION_SIZE + TRAIN_SIZE)) * num_parts:
            validation_loader.append(batch)
        else:
            train_loader.append(batch)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    flower_client = FlowerClient(
        net, train_loader, validation_loader, test_loader,
    )
    flower_client.set_context(context)
    return flower_client.to_client()


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
    # mods=[
    #     secaggplus_mod,  # Comment-out to disable SecAgg+
    #     local_dp_obj  # Comment-out to disable DP
    # ],
)
