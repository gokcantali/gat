# Run simulation
from flwr.simulation import run_simulation

from fl.client import client
from fl.server import server, backend_config

run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=3,
    backend_config=backend_config,
)
