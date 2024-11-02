# fedl: A Flower / PyTorch app

## Install dependencies and project

The dependencies must be installed by using the _pyproject.toml_ file in the **root** folder of the project.
You can use either pip or poetry:

```bash
pip install -e .
```

```bash
poetry install
```

## Check the Flower Architecture

To gain a better understand on how Flower handles server and client sides,
you can take a look at the following documentation that explains the Flower architecture:

[Flower Architecture](https://flower.ai/docs/framework/explanation-flower-architecture.html)

## Run the ML task with Federated Learning (FL)

To successfully run the FL part, you will need to follow multiple steps:

### Start SuperLink on the Server Side

First, you should spin up the long-living server-app process which is called SuperLink:

```bash
flower-superlink --insecure
```

NOTE: You may need to add `poetry run` prefix to the commands, if you use Poetry instead of pip.

### Start multiple SuperNode instances on the Client Side

Now, you should instantiate multiple SuperNode instances, which are long-living client-app processes.
For each client you want to use, you can execute the following command on a separate shell session,
after replacing the `<node-id>` with the actual _integer_ value for that client's ID:

```bash
flower-supernode fedl --server 0.0.0.0:9092 --insecure --node-config 'partition-id=<node-id>'
```

For instance, if you want to start 3 clients, you can run the following commands,
where each of them is executed in a separate shell session:

```bash
flower-supernode fedl --server 0.0.0.0:9092 --insecure --node-config 'partition-id=0'
flower-supernode fedl --server 0.0.0.0:9092 --insecure --node-config 'partition-id=1'
flower-supernode fedl --server 0.0.0.0:9092 --insecure --node-config 'partition-id=2'
```

NOTE: You may need to add `poetry run` prefix to the commands, if you use Poetry instead of pip.

### Start the Aggregator on the Server Side

As the final step, you can start the server aggregator,
which will trigger the execution of client app on each Supernode (i.e. each FL Client),
using the following command:

```bash
flower-server-app fedl --superlink 0.0.0.0:9091 --insecure
```

NOTE: You may need to add `poetry run` prefix to the commands, if you use Poetry instead of pip.

Now, you should be able to see the logs on each running client app.
