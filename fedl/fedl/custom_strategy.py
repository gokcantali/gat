import random
from logging import INFO, WARNING
from typing import Optional

import numpy as np
from flwr.common import FitIns, Parameters, log, GetPropertiesIns
from flwr.server import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy import FedAvg


class SimpleClientManagerWithCustomSampling(SimpleClientManager):
    def sample(
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
        weights = []
        log(WARNING, "Clients with green energy:")
        for cid in available_cids:
            green_energy = random.choice([0, 0, 1])
            if green_energy:
                log(WARNING, cid)
                weights.append(10)
            else:
                weights.append(1)

        sampled_cids = np.random.choice(
            available_cids, size=num_clients,
            replace=False, p=weights/np.sum(weights)
        )

        log(WARNING, "Here are the selected clients:")
        for cid in sampled_cids:
            log(WARNING, cid)
        return [self.clients[cid] for cid in sampled_cids]


class CustomStrategy(FedAvg):
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: SimpleClientManagerWithCustomSampling
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
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
