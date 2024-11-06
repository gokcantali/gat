# Create FedAvg strategy
from typing import List, Tuple

from flwr.common import Context, Metrics, Parameters
from flwr.server import ServerAppComponents, ServerConfig, ServerApp, start_server, Driver, LegacyContext, ClientManager
from flwr.server.strategy import FedAvg
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from .custom_strategy import SimpleClientManagerWithCustomSampling


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


strategy = FedAvg(
    fraction_fit=0.6,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=3,  # Never sample less than 3 clients for training
    min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
    min_available_clients=3,  # Wait until all 3 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # Use weighted average as custom metric evaluation function
)

# Configure the server for 5 rounds of training
config = ServerConfig(num_rounds=5)


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    return ServerAppComponents(
        strategy=strategy, config=config,
        client_manager=SimpleClientManagerWithCustomSampling()
    )


# Create the ServerApp
app = ServerApp(server_fn=server_fn)


### Comment-out the following code block ###
### to disable SecAgg+ Secure Aggregation ###
@app.main()
def main(driver: Driver, context: Context) -> None:
    # Construct the LegacyContext
    num_rounds = context.run_config["num-server-rounds"]
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_manager=SimpleClientManagerWithCustomSampling(),
    )

    fit_workflow = SecAggPlusWorkflow(
        num_shares=context.run_config["num-shares"],
        reconstruction_threshold=context.run_config["reconstruction-threshold"],
        max_weight=context.run_config["max-weight"],
    )

    # Create the workflow
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    # Execute
    workflow(driver, context)
