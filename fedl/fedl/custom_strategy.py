import random
from logging import INFO, WARNING
from typing import Optional

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

        sampled_cids = random.sample(available_cids, num_clients)
        log(WARNING, "Here are the clients:::")
        for cid in self.clients:
            log(WARNING, self.clients[cid])
            log(WARNING, self.clients[cid].properties)
            # log(WARNING, self.clients[cid].get_properties(
            #     ins=GetPropertiesIns(config={}), timeout=None, group_id=None
            # ))
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
