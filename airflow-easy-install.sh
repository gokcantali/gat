# For non-Linux systems, disable the following command
sudo apt install -y python3-dev python3.11-dev \
  build-essential libssl-dev libffi-dev \
  libxml2-dev libxslt1-dev zlib1g-dev

poetry run pip install "apache-airflow[celery]==2.10.5" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt"

poetry run pip install "setproctitle<1.3"

mkdir -p data/subsample

sed -i "s/\/Users\/canl\/Projects\/BA\-GNN\/gat/\/home\/ubuntu\/gat/g" airflow.cfg

echo "SOURCE_DATASET=worker$1-traces-75min.csv" > .env

export AIRFLOW_HOME="$(pwd)"

echo "export AIRFLOW_HOME=/home/ubuntu/gat" >> ~/.bashrc

nohup poetry run airflow standalone > nohup-airflow.txt 2>&1 &
