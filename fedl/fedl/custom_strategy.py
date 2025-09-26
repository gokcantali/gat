import math
# import random
import traceback
from logging import INFO, WARNING, ERROR
from typing import Optional, Union, Dict

import numpy as np
import pandas as pd
from flwr.common import FitIns, Parameters, log, FitRes, Scalar, EvaluateRes, parameters_to_ndarrays, GetPropertiesIns
from flwr.server import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy import FedProx
from sklearn.linear_model import LinearRegression


CF_METHODS = {
    "simple_avg": "CF_SimpleAvg",
    "exp_smooth": "CF_ExpSmooth",
    "lin_reg": "CF_LinRegress",
    "non_cf": "NON_CF"
}


def exponential_smoothing(data, alpha):
    """
    Apply simple exponential smoothing to a time-series data.

    Parameters:
    - data: List of numerical values representing the time-series.
    - alpha: Smoothing factor, a float between 0 and 1.

    Returns:
    - smoothed_data: List of smoothed values.
    """
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be a value between 0 and 1.")

    if len(data) == 0:
        raise ValueError("Input data must not be empty.")

    smoothed_data = [data[0]]  # First value is same as the original

    for t in range(1, len(data)):
        smoothed_value = alpha * data[t] + (1 - alpha) * smoothed_data[t-1]
        smoothed_data.append(smoothed_value)

    return smoothed_data


class SimpleClientManagerWithPrioritizedSampling(SimpleClientManager):
    def sample_with_priority(
        self,
        num_clients: int,
        priorities: dict[str, float],
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> list[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        log(WARNING, f"Priorities: {priorities}")

        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        prior_weights = []
        sum_weights = 0.0
        for cid in available_cids:
            priority = priorities.get(cid, -1)
            prior_weights.append(priority)
            if priority >= 0:
                sum_weights += priority
        for ind in range(len(prior_weights)):
            if prior_weights[ind] <= 0.0:
                prior_weights[ind] = sum_weights / len(prior_weights)

        prior_weights /= np.sum(prior_weights)

        sampled_cids = np.random.choice(
            available_cids, size=num_clients, replace=False, p=prior_weights
        )

        log(WARNING, f"Priority weights: {prior_weights}")
        log(WARNING, "Here are the selected clients:")
        for cid in sampled_cids:
            log(WARNING, cid)
        return [self.clients[cid] for cid in sampled_cids]


class FedAvgCF(FedProx):
    def __init__(
        self, alpha: float, window: int,
        method: str = "lin_reg", total_rounds: int = 60,
        trial: str = "First",
        log_params_and_metrics_fn: Optional[callable] = None,
        proximal_mu: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs, proximal_mu=proximal_mu)
        self.alpha = alpha
        self.window = window
        self.cid_to_pid: Dict[str, str] = {}  # server-side cache: ClientProxy.cid -> partition-id

        if method not in CF_METHODS:
            raise Exception(f"Method: {method} not implemented!")
        self.method = method

        self.total_rounds = total_rounds
        self.trial = trial
        self.log_params_and_metrics_fn = log_params_and_metrics_fn

        # emission mapping keyed by partition-id (when available), otherwise fallback to client cid
        self.emission_mapping: dict[str, dict[int, float]] = {}

        # store selected participants per round as list of partition ids (or fallbacks)
        self.selected_participants_per_round: dict[int, list[str]] = {}

    def _prime_partition_map(self, client_manager: SimpleClientManagerWithPrioritizedSampling) -> None:
        # Ask any clients we haven't seen yet for their properties
        for cid, client in client_manager.all().items():
            if cid in self.cid_to_pid:
                continue
            res = client.get_properties(
                ins=GetPropertiesIns(config={}),
                timeout=20,
                group_id=None
            )

            # On the server, this returns a GetPropertiesRes-like object with .properties in Classic API
            props = getattr(res, "properties", res)  # support both wrappers and raw dict
            self.cid_to_pid[cid] = str(props["partition-id"])

    def initialize_parameters(self, client_manager: SimpleClientManagerWithPrioritizedSampling) -> Parameters | None:
        # Optional: prime once before round 1
        initial_params = super().initialize_parameters(client_manager)

        client_manager.wait_for(num_clients=5, timeout=20)
        self._prime_partition_map(client_manager)
        print(f"Client to Partition mapping: {self.cid_to_pid}")

        return initial_params

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        loss_agg, metrics_agg = super().aggregate_evaluate(
            server_round=server_round, results=results, failures=failures
        )

        if self.log_params_and_metrics_fn:
            self.log_params_and_metrics_fn(None, metrics_agg)

        return loss_agg, metrics_agg

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: SimpleClientManagerWithPrioritizedSampling,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        else:
            # Default fit config function
            config = {
                "cf_method": CF_METHODS[self.method],
                "total_rounds": self.total_rounds,
                "trial": self.trial,
            }
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        if server_round <= self.window:
            log(INFO, f"Round: {server_round} <= Window: {self.window} - Skipping...")
            clients = client_manager.sample(
                num_clients=sample_size,
                min_num_clients=min_num_clients,
            )
        else:
            if self.method == "lin_reg":
                priorities = self._calculate_carbon_based_priorities_using_linear_regression(
                    server_round=server_round
                )
            elif self.method == "exp_smooth":
                priorities = self._calculate_carbon_based_priorities_using_smoothing(
                    server_round=server_round
                )
            elif self.method == "simple_avg":
                priorities = self._calculate_carbon_based_priorities()
            else:  # NON_CF
                log(WARNING, f"Carbon Reduction disabled! Fallback to standard sampling...")
                priorities = {}

            # if priorities are empty, fallback to standard sampling
            if len(priorities.keys()) == 0 or np.sum(list(priorities.values())) <= 0.0:
                clients = client_manager.sample(
                    num_clients=sample_size,
                    min_num_clients=min_num_clients,
                )
            else:
                try:
                    # handles errors that might occur during the prioritized sampling
                    clients = client_manager.sample_with_priority(
                        num_clients=sample_size,
                        priorities=priorities,
                        min_num_clients=min_num_clients,
                    )
                except Exception as _:
                    log(ERROR, traceback.format_exc())

                    # fallback to the simple sampling
                    clients = client_manager.sample(
                        num_clients=sample_size,
                        min_num_clients=min_num_clients,
                    )

        # Record the selected participants using partition-id if available
        selected_partition_ids = []
        for client in clients:
            partition_id = self.cid_to_pid[client.cid]
            selected_partition_ids.append(partition_id)

        # store in-memory mapping
        self.selected_participants_per_round[server_round] = selected_partition_ids

        # append to CSV for easy inspection / external logging
        try:
            csv_path = f"selected_participants_{self.method}_{self.trial}.csv"
            with open(csv_path, "a") as f:
                # write header if file newly created
                # append line: round,partition1;partition2;...
                f.write(f"{server_round},{','.join(map(str, selected_partition_ids))}\n")
        except Exception:
            log(WARNING, "Failed to persist selected participants to CSV.")

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        for client_proxy, fit_res in results:
            # use partition id (preferred) as the key in emissions mapping
            cid = getattr(client_proxy, "cid", None)
            if cid not in self.emission_mapping:
                self.emission_mapping[cid] = {}
            carbon_emission = fit_res.metrics.get("carbon", -1.0)
            self.emission_mapping[cid][server_round] = carbon_emission

        params_agg, metrics_agg = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )

        if self.log_params_and_metrics_fn:
            self.log_params_and_metrics_fn(
                parameters_to_ndarrays(params_agg), metrics_agg
            )

        return params_agg, metrics_agg

    def _calculate_carbon_based_priorities(self):
        priorities = {}
        mean_emission_total = 0.0
        for cid, emissions in self.emission_mapping.items():
            if np.mean(list(emissions.values())) < 0:
                priorities[cid] = -1
            else:
                priorities[cid] = np.mean(
                    list(filter(lambda e: e > 0, list(emissions.values())))
                )
            mean_emission_total += priorities[cid]

        if mean_emission_total <= 0.0:
            for cid in priorities:
                priorities[cid] = 1 / len(self.emission_mapping.keys())
        else:
            for cid in priorities:
                if priorities[cid] > 0:
                    priorities[cid] = mean_emission_total / priorities[cid]
                else:
                    priorities[cid] = mean_emission_total / len(
                        self.emission_mapping.keys()
                    )

        return priorities

    def _calculate_carbon_based_priorities_using_smoothing(
        self,
        server_round: int,
    ):
        """uses a simple exponential smoothing logic for estimating
        the next round carbon emission and determine the priorities"""

        # calculate the mean emission per round across clients
        mean_emission_per_round = {}
        for fl_round in range(server_round - self.window, server_round, 1):
            measurement_count = 0
            total_emission_in_round = 0.0
            for _, emissions in self.emission_mapping.items():
                emission = emissions.get(fl_round, -1)
                if not emission or emission <= 0 or math.isnan(emission):
                    continue
                measurement_count += 1
                total_emission_in_round += emission
            if measurement_count != 0:
                mean_emission_per_round[fl_round] = (
                    total_emission_in_round / measurement_count
                )

        # if no measurement across any rounds of the time window,
        # return equal priorities
        if len(mean_emission_per_round.keys()) == 0:
            return {cid: 1 for cid in self.emission_mapping.keys()}

        # if there is non-measured rounds, take the mean across
        # measurements across available rounds
        for fl_round in range(server_round - self.window, server_round, 1):
            if fl_round not in mean_emission_per_round:
                mean_emission_per_round[fl_round] = np.mean(
                    list(mean_emission_per_round.values())
                )

        # compute the emission estimations for the next round
        next_round_emission_estimations = {}
        for cid, emissions in self.emission_mapping.items():
            emission_list = []

            for fl_round in range(server_round - self.window, server_round, 1):
                emission = emissions.get(fl_round, -1)
                if not emission or emission <= 0 or math.isnan(emission):
                    emission = mean_emission_per_round[fl_round]
                emission_list.append(emission)

            smoothed_emissions = exponential_smoothing(emission_list, self.alpha)

            next_round_emission_estimations[cid] = smoothed_emissions[-1]

        log(WARNING, f"Next Round Estimations: {next_round_emission_estimations}")
        mean_estimation_for_next_round = np.mean(
            list(next_round_emission_estimations.values())
        )

        # calculate and return the priorities based on the proportion of the mean estimation
        # to each of the client's estimation
        return {
            cid: mean_estimation_for_next_round / next_round_emission_estimations[cid]
            for cid in self.emission_mapping.keys()
        }

    def _calculate_carbon_based_priorities_using_linear_regression(
        self,
        server_round: int,
    ):
        """uses a simple Linear Regression model for estimating
        the next round carbon emission and determine the priorities"""

        # calculate the mean emission per round across clients
        mean_emission_per_round = {}
        for fl_round in range(1, server_round, 1):
            measurement_count = 0
            total_emission_in_round = 0.0
            for _, emissions in self.emission_mapping.items():
                emission = emissions.get(fl_round, -1)
                if not emission or emission <= 0 or math.isnan(emission):
                    continue
                measurement_count += 1
                total_emission_in_round += emission
            if measurement_count != 0:
                mean_emission_per_round[fl_round] = (
                    total_emission_in_round / measurement_count
                )

        # if no measurement across any rounds of the time window,
        # return equal priorities
        if len(mean_emission_per_round.keys()) == 0:
            return {cid: 1 for cid in self.emission_mapping.keys()}

        # if there is non-measured rounds, take the mean across
        # measurements across available rounds
        for fl_round in range(1, server_round, 1):
            if fl_round not in mean_emission_per_round:
                mean_emission_per_round[fl_round] = np.mean(
                    list(mean_emission_per_round.values())
                )

        # compute the emission estimations for the next round
        next_round_emission_estimations = {}
        for cid, emissions in self.emission_mapping.items():
            emission_values_for_regression = []
            rounds_for_regression = []

            for fl_round in range(1, server_round, 1):
                emission = emissions.get(fl_round, 0)
                if not emission or emission <= 0 or math.isnan(emission):
                    continue
                rounds_for_regression.append(fl_round)
                emission_values_for_regression.append(emission)

            if len(rounds_for_regression) < self.window:
                next_round_emission_estimations[cid] = mean_emission_per_round[server_round-1]
            else:
                lin_reg_model = LinearRegression()
                lin_reg_model.fit(
                    pd.DataFrame(rounds_for_regression), emission_values_for_regression
                )
                next_round_emission_estimations[cid] = (
                    lin_reg_model.predict(pd.DataFrame([server_round]))[0]
                )

        log(WARNING, f"Next Round Estimations: {next_round_emission_estimations}")
        mean_estimation_for_next_round = np.mean(
            list(next_round_emission_estimations.values())
        )

        # calculate and return the priorities based on the proportion of the mean estimation
        # to each of the client's estimation
        return {
            cid: mean_estimation_for_next_round / next_round_emission_estimations[cid]
            for cid in self.emission_mapping.keys()
        }
