import os
import textwrap
from datetime import datetime, timedelta

import dotenv
# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from dotenv import load_dotenv

from gat.converter import create_one_graph_from_the_first_existing_dataset_subsample
from gat.load_data import create_subset_from_dataset_using_monte_carlo

load_dotenv()

with DAG(
    "dataset_sampling",
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        "depends_on_past": False,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'sla': timedelta(hours=2),
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function, # or list of functions
        # 'on_success_callback': some_other_function, # or list of functions
        # 'on_retry_callback': another_function, # or list of functions
        # 'sla_miss_callback': yet_another_function, # or list of functions
        # 'on_skipped_callback': another_function, #or list of functions
        # 'trigger_rule': 'all_success',
    },
    description="One-task DAG for sampling dataset",
    schedule="0 * * * *",
    start_date=datetime(2025, 3, 7, 0, 0),
    end_date=datetime(2025, 3, 23, 4, 0),
    catchup=False,
    tags=["example"],
    params={
        "dataset_file_name": os.environ.get(
            "SOURCE_DATASET", "sampled-traces-3ddos-2zap-1scan.csv"
        ),
    },
    is_paused_upon_creation=False
) as dag:

    t1 = PythonOperator(
        task_id="sample_dataset",
        depends_on_past=False,
        python_callable=create_subset_from_dataset_using_monte_carlo,
        op_args=[
            "{{ params.dataset_file_name }}",
        ]
    )

    t2 = PythonOperator(
        task_id="graph_construction",
        depends_on_past=False,
        python_callable=create_one_graph_from_the_first_existing_dataset_subsample,
    )

    t1 >> t2
