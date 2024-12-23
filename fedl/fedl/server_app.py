# Create FedAvg strategy
from typing import List, Tuple

from flwr.common import Context, Metrics, Parameters
from flwr.server import ServerAppComponents, ServerConfig, ServerApp, start_server, Driver, LegacyContext, ClientManager
from flwr.server.strategy import FedAvg, FedProx
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from .custom_strategy import SimpleClientManagerWithPrioritizedSampling, FedAvgCF


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


def report_carbon_emissions(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_emission = 0.0
    for num_examples, m in metrics:
        print(f"Client with {num_examples} samples emitted {m['carbon']} kgCO2")
        total_emission += m["carbon"]
    with open("carbon_emissions.txt", "a") as f:
        f.write(f"{total_emission}\n")
    return {"total_emission": total_emission}


strategy = FedAvg(
    fraction_fit=0.6,  # Sample 60% of available clients for training
    fraction_evaluate=1,  # Sample 100% of available clients for evaluation
    min_fit_clients=3,  # Never sample less than 3 clients for training
    min_evaluate_clients=3,  # Never sample less than 3 clients for evaluation
    min_available_clients=3,  # Wait until all 3 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # Use weighted average as custom metric evaluation function
    fit_metrics_aggregation_fn=report_carbon_emissions,  # Use custom function to report carbon emissions
)

# Configure the server for 60 rounds of training
config = ServerConfig(num_rounds=60)


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    return ServerAppComponents(
        config=config,
        strategy=strategy,
        # client_manager=SimpleClientManagerWithPrioritizedSampling()
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
