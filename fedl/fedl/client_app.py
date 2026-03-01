import math
import time
from dataclasses import asdict
from pathlib import Path

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
from vnf_ds_benchmark import get_sequences

TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10
TRAIN_SIZE = 1 - VALIDATION_SIZE - TEST_SIZE
RNN_CELL = "LSTM"  # or "LSTM" or "RNN


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, testloader, **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        pid = self.get_context().node_config["partition-id"]
        return {"partition-id": pid}

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
        self.net.set_parameters(parameters, config, is_evaluate=False)
        tracker.start()
        metrics = self.net.train_model(self.trainloader, self.valloader, batch_mode=True, epochs=1)
        emissions = tracker.stop()
        # self.emissions = emissions if not math.isnan(emissions) else emissions

        metrics_to_aggregate = {
            "carbon": emissions
        }
        for metric, values in asdict(metrics).items():
            metrics_to_aggregate[metric] = values[-1]

        # Include hyperparameters in the metrics to aggregate
        metrics_to_aggregate["learning_rate"] = self.net.scheduler.get_last_lr()[0]
        for initial_hp, hp_value in self.net.hyperparams.items():
            metrics_to_aggregate[f"hp:{initial_hp}"] = hp_value
        metrics_to_aggregate["hp:epochs"] = 1

        return (
            self.net.get_parameters(),
            len(self.trainloader),
            metrics_to_aggregate
        )

    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters, config, is_evaluate=True)
        _, loss, perf_metrics = self.net.test_model_batch_mode(self.testloader)
        print("METRICS OF CLIENT:")
        print(perf_metrics)
        return loss, len(self.testloader), perf_metrics


class FlowerClientRNN(NumPyClient):
    def __init__(
        self, net,
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        X_test_seq, y_test_seq,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.net = net
        self.X_train_seq = X_train_seq
        self.y_train_seq = y_train_seq
        self.X_val_seq = X_val_seq
        self.y_val_seq = y_val_seq
        self.X_test_seq = X_test_seq
        self.y_test_seq = y_test_seq

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        pid = self.get_context().node_config["partition-id"]
        print("Get Properties called. Partition ID: ", pid)
        return {"partition-id": pid}

    def get_parameters(self, config):
        return self.net.get_parameters()

    def fit(self, parameters, config):
        from vnf_ds_benchmark import train_model as train_model_rnn
        tracker = EmissionsTracker(
            measure_power_secs=10,
            experiment_id="6d102989-535b-459e-ab34-406ee6a2bb54",
            save_to_api=False,
            allow_multiple_runs=True
        )
        self.net.set_parameters(parameters, config, is_evaluate=False)
        tracker.start()
        start_time_train_rnn = time.time()
        new_model, loss, f1_score = train_model_rnn(
            self.net, self.X_train_seq, self.y_train_seq, self.X_val_seq, self.y_val_seq,
            rnn_cell=RNN_CELL
        )
        end_time_train_rnn = time.time()
        print("Client Training Time: ", end_time_train_rnn - start_time_train_rnn)
        emissions = tracker.stop()
        # self.emissions = emissions if not math.isnan(emissions) else emissions
        print("TRAINING DONE!")
        print("F1 Score:", f1_score)
        f1_score = f1_score[0] if isinstance(f1_score, tuple) else f1_score

        metrics_to_aggregate = {
            "training_f1_score": f1_score,
            "carbon": emissions,
            "validation_loss": loss
        }
        print("METRICS OF CLIENT: ", metrics_to_aggregate)

        return (
            new_model.get_parameters(),
            len(self.X_train_seq),
            metrics_to_aggregate
        )

    def evaluate(self, parameters, config):
        from vnf_ds_benchmark import evaluate_model as evaluate_model_rnn

        self.net.set_parameters(parameters, config, is_evaluate=True)
        loss, f1, report = evaluate_model_rnn(self.net, self.X_test_seq, self.y_test_seq)

        # report now contains: report["auroc"], report["pr_auc"], report["recall_at_1_fpr"]
        metrics = {
            "testing_f1_score": float(f1),
            "testing_auroc": float(report.get("auroc", float("nan"))),
            "testing_pr_auc": float(report.get("pr_auc", float("nan"))),
            "testing_recall_at_1fpr": float(report.get("recall_at_1_fpr", float("nan"))),
        }

        print("METRICS OF CLIENT:")
        print(metrics)
        return loss, len(self.X_test_seq), metrics


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


def construct_flower_client_rnn_vnf_datasets(client_id, context):
    from vnf_ds_benchmark import prepare_sequences, initialize_model
    print("Constructing Flower Client for VNF dataset - with client id: ", client_id)

    net = initialize_model(cell_type=RNN_CELL)

    df_dict = get_sequences(dataset_id=client_id)
    all_df_dict = get_sequences(dataset_id=-1) # sequences of ALL datasets
    # check if datasets already exist
    if df_dict is None:
        # prepare datasets
        start_time_prepare_ds = time.time()
        df_dict = prepare_sequences(
            dataset_ids=[client_id],
        )
        end_time_prepare_ds = time.time()
        print("Dataset Preparation Time: ", end_time_prepare_ds - start_time_prepare_ds)

    flower_client = FlowerClientRNN(
        net,
        df_dict['X_train'],
        df_dict['y_train'],
        df_dict['X_val'],
        df_dict['y_val'],
        all_df_dict['X_test'] if all_df_dict is not None else df_dict['X_test'],
        all_df_dict['y_test'] if all_df_dict is not None else df_dict['y_test']
    )
    flower_client.set_context(context)
    return flower_client.to_client()


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    partition_id = context.node_config["partition-id"]

    # Construct the client
    # flower_client = construct_flower_client_rnn_vnf_datasets(
    #     client_id=partition_id, context=context
    # )
    flower_client = construct_flower_client(
        client_id=partition_id, context=context
    )
    return flower_client


# Create an instance of the mod with the required params
local_dp_obj = LocalDpMod(
    0.9, 0.8, 0.01, 0.02
)

# Create the ClientApp
app = ClientApp(
    client_fn=client_fn,
    # mods=[
    #     #secaggplus_mod,  # Comment-out to disable SecAgg+
    #     local_dp_obj  # Comment-out to disable DP
    # ],
)
