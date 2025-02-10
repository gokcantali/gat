# Create FedAvg strategy
from typing import List, Tuple, Optional

import mlflow
from flwr.common import Context, Metrics, Parameters, Scalar, NDArray, NDArrays
from flwr.server import ServerAppComponents, ServerConfig, ServerApp, start_server, Driver, LegacyContext, ClientManager
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from run import initialize_gcn_model
from .custom_strategy import SimpleClientManagerWithPrioritizedSampling, FedAvgCF, CF_METHODS

mlflow.set_tracking_uri("http://localhost:8080")

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    aggregated_metrics = {}
    total_examples = 0

    for num_examples, m in metrics:
        total_examples += num_examples
        for key, value in m.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0

            # multiply metric of each client by number of examples used
            aggregated_metrics[key] += num_examples * value

    # divide by total number of examples to get weighted average
    for metric in aggregated_metrics:
        aggregated_metrics[metric] /= total_examples

    # return aggregated metrics in the weighted average form
    return aggregated_metrics


NUM_ROUNDS = 3
METHOD = "simple_avg"
TRIAL = "23th"

EXPERIMENT_NAME = f"5Nodes-{NUM_ROUNDS}Rounds-{METHOD}"
EXPERIMENT_ID = mlflow.set_experiment(
    experiment_name=EXPERIMENT_NAME
).experiment_id

RUN_ID = mlflow.start_run(
    experiment_id=EXPERIMENT_ID,
    run_name=f"{TRIAL} Trial"
).info.run_id
mlflow.end_run()

current_training_round = 0

def training_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global current_training_round
    current_training_round += 1

    total_emission = 0.0
    total_samples = 0
    other_metrics_weighted_total = {}

    for num_examples, m in metrics:
        print(f"Client with {num_examples} samples emitted {m['carbon']} kgCO2")
        total_emission += m["carbon"]
        for field in m:
            if field != "carbon":
                if field not in other_metrics_weighted_total:
                    other_metrics_weighted_total[field] = 0.0
                other_metrics_weighted_total[field] += num_examples * m[field]
        total_samples += num_examples

    with open("carbon_emissions.txt", "a") as f:
        f.write(f"{total_emission}\n")

    other_metrics_weighted_average = {
        field: other_metrics_weighted_total[field] / total_samples
        for field in other_metrics_weighted_total
    }

    return {"total_emission": total_emission, **other_metrics_weighted_average}


def log_model_params_and_metrics_to_mlflow(
    params: Optional[NDArrays] = None,
    metrics_plus_hyperparams: Optional[dict[str, Scalar]] = None
):
    global EXPERIMENT_NAME, RUN_ID, NUM_ROUNDS, TRIAL, current_training_round
    with mlflow.start_run(run_id=RUN_ID):
        if metrics_plus_hyperparams:
            hyperparam_prefix = "hp:"

            hyperparams = {}
            metrics = {}
            for metric, value in metrics_plus_hyperparams.items():
                if metric.startswith(hyperparam_prefix):
                    hyperparams[metric[len(hyperparam_prefix):]] = value
                else:
                    metrics[metric] = value

            mlflow.log_metrics(metrics, step=current_training_round)
            if current_training_round == 1:
                # log the hyperparams only once at the beginning
                mlflow.log_params(hyperparams)

        if params and current_training_round == NUM_ROUNDS:
            # log the model only once at the end
            model = initialize_gcn_model(num_classes=4)
            model.set_parameters(params, None, False)
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=f"GNN-{EXPERIMENT_NAME}",
                registered_model_name=f"GNN-{EXPERIMENT_NAME}-{TRIAL}Trial"
            )

strategy = FedAvgCF(
    fraction_fit=0.6,  # Sample 60% of available clients for training
    fraction_evaluate=1,  # Sample 100% of available clients for evaluation
    min_fit_clients=3,  # Never sample less than 3 clients for training
    min_evaluate_clients=3,  # Never sample less than 3 clients for evaluation
    min_available_clients=3,  # Wait until all 3 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # Use weighted average as custom metric evaluation function
    fit_metrics_aggregation_fn=training_metrics_aggregation,  # Use custom function to report carbon emissions
    alpha=0.5,
    window=5,
    method=METHOD,
    trial=TRIAL,
    total_rounds=NUM_ROUNDS,
    log_params_and_metrics_fn=log_model_params_and_metrics_to_mlflow
)

# Configure the server for <NUM_ROUNDS> rounds of training
config = ServerConfig(num_rounds=NUM_ROUNDS)


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    global NUM_ROUNDS, TRIAL, METHOD

    # add a new subheader in the carbon emission file
    # before the next simulation starts
    with open("carbon_emissions.txt", "a") as f:
        subheader = "\n====== "
        subheader += "WITHOUT OPTIMIZATION" if METHOD == 'non_cf' else "WITH OPTIMIZATION"
        subheader += f" - {NUM_ROUNDS} Rounds - {CF_METHODS[METHOD]} Algorithm - {TRIAL} TRIAL"
        subheader += " - CLOUD ======\n"
        f.write(subheader)

    return ServerAppComponents(
        config=config,
        strategy=strategy,
        client_manager=SimpleClientManagerWithPrioritizedSampling()
    )


# Create the ServerApp
app = ServerApp(server_fn=server_fn)


### Comment-out the following code block ###
### to disable SecAgg+ Secure Aggregation ###
# @app.main()
# def main(driver: Driver, context: Context) -> None:
#     # Construct the LegacyContext
#     num_rounds = context.run_config["num-server-rounds"]
#     context = LegacyContext(
#         context=context,
#         config=ServerConfig(num_rounds=num_rounds),
#         strategy=strategy,
#         client_manager=SimpleClientManagerWithCustomSampling(),
#     )
#
#     fit_workflow = SecAggPlusWorkflow(
#         num_shares=context.run_config["num-shares"],
#         reconstruction_threshold=context.run_config["reconstruction-threshold"],
#         max_weight=context.run_config["max-weight"],
#     )
#
#     # Create the workflow
#     workflow = DefaultWorkflow(fit_workflow=fit_workflow)
#
#     # Execute
#     workflow(driver, context)
