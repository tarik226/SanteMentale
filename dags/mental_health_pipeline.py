
from datetime import timedelta
import logging
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

from tasks.prepare_data import run as prepare_data
from tasks.preprocess import run as preprocess_data
from tasks.train_models import run as train_models
from tasks.evaluate import run as evaluate_models


# Error monitoring
def task_failure_callback(context):

    task_instance = context["task_instance"]

    exception = context.get("exception")

    logging.error(
        "TASK FAILED: %s | DAG: %s | ERROR: %s",
        task_instance.task_id,
        task_instance.dag_id,
        exception
    )


# Default configuration
default_args = {
    "owner": "airflow",

    "depends_on_past": False,

    "on_failure_callback": task_failure_callback
}


with DAG(

    dag_id="mental_health_ml_pipeline",

    description="Mental Health Risk Prediction ML Pipeline",

    default_args=default_args,

    start_date=pendulum.datetime(
        2026, 1, 1,
        tz="Africa/Casablanca"
    ),

    schedule=None,

    catchup=False,

    max_active_runs=1,

    tags=[
        "machine-learning",
        "mental-health",
        "classification"
    ]

) as dag:

    # Task 1: Data preparation
    preparation = PythonOperator(

        task_id="prepare_data",

        python_callable=prepare_data

    )

    # Task 2: Data preprocessing
    preprocessing = PythonOperator(

        task_id="preprocess_data",

        python_callable=preprocess_data

    )

    # Task 3: Model training
    training = PythonOperator(

        task_id="train_models",

        python_callable=train_models

    )

    # Task 4: Model evaluation
    evaluation = PythonOperator(

        task_id="evaluate_models",

        python_callable=evaluate_models

    )

    # Task dependencies
    preparation >> preprocessing >> training >> evaluation
