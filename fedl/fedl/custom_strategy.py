import random
from logging import INFO, WARNING
from typing import Optional, Union

import numpy as np
from flwr.common import FitIns, Parameters, log, GetPropertiesIns, FitRes, Scalar
from flwr.server import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy import FedAvg


class SimpleClientManagerWithCustomSampling(SimpleClientManager):
    def sample_with_probability(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> list[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
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

        # apply green energy prioritization logic
        # assume that clients have a property called "green_energy" which is a boolean
        # for now, we will assign this property randomly
        # weights = []
        # log(WARNING, "Clients with green energy:")
        # for cid in available_cids:
        #     green_energy = random.choice([0, 0, 1])
        #     if green_energy:
        #         log(WARNING, cid)
        #         weights.append(10)
        #     else:
        #         weights.append(1)
        #
        # sampled_cids = np.random.choice(
        #     available_cids, size=num_clients,
        #     replace=False, p=weights/np.sum(weights)
        # )
        sampled_cids = np.random.choice(
            available_cids, size=num_clients, replace=False
        )

        log(WARNING, "Here are the selected clients:")
        for cid in sampled_cids:
            log(WARNING, cid)
            client_props = self.clients[cid].get_properties(
                ins=GetPropertiesIns(config={"emissions": True}),
                timeout=30,
                group_id=None
            )
            log(WARNING, f"Properties: {client_props}")
        return [self.clients[cid] for cid in sampled_cids]


class SimpleClientManagerWithPrioritizedSampling(SimpleClientManager):
    def sample_with_priority(
        self,
        num_clients: int,
        priorities: dict[str, float],
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,

    ) -> list[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # if priorities are empty, fallback to standard sampling
        if len(priorities.keys()) == 0 or np.sum(list(priorities.values())) <= 0.0:
            return super().sample(
                num_clients=num_clients,
                min_num_clients=min_num_clients,
                criterion=criterion,
            )

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
            available_cids, size=num_clients,
            replace=False, p=prior_weights
        )

        log(WARNING, f"Priority weights: {prior_weights}")
        log(WARNING, "Here are the selected clients:")
        for cid in sampled_cids:
            log(WARNING, cid)
        return [self.clients[cid] for cid in sampled_cids]

class FedAvgCF(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.emission_mapping: dict[str, list[float]] = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: SimpleClientManagerWithPrioritizedSampling
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        priorities = self._calculate_carbon_based_priorities()

        clients = client_manager.sample_with_priority(
            num_clients=sample_size,
            priorities=priorities,
            min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            if cid not in self.emission_mapping:
                self.emission_mapping[cid] = []
            self.emission_mapping[cid].append(
                fit_res.metrics.get("carbon", -1.0)
            )

        log(WARNING, f"Carbon Emissions: {self.emission_mapping}")

        return super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures
        )

    def _calculate_carbon_based_priorities(self):
        priorities = {}
        mean_emission_total = 0.0
        for cid, emissions in self.emission_mapping.items():
            if np.mean(emissions) < 0:
                priorities[cid] = -1
            else:
                priorities[cid] = np.mean(list(filter(lambda e: e > 0, emissions)))
            mean_emission_total += priorities[cid]

        if mean_emission_total <= 0.0:
            for cid in priorities:
                priorities[cid] = 1 / len(self.emission_mapping.keys())
        else:
            for cid in priorities:
                if priorities[cid] > 0:
                    priorities[cid] = mean_emission_total / priorities[cid]
                else:
                    priorities[cid] = mean_emission_total / len(self.emission_mapping.keys())

        return priorities
