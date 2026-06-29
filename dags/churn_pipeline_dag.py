"""
Airflow DAG: orchestrates the churn prediction pipeline.

validate -> train

Each task shells out to the same scripts you've already run manually
and through `dvc repro`. Airflow's job here is purely orchestration --
scheduling, retries, and a visual run history -- not reimplementing
any pipeline logic.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

# Working directory inside the Airflow containers where src/ and data/
# are volume-mounted (see docker-compose.yaml)
PROJECT_DIR = "/opt/airflow"

default_args = {
    "owner": "abishek",
    "retries": 1,
}

with DAG(
    dag_id="churn_prediction_pipeline",
    description="Validate churn data, then train and log a model to MLflow",
    default_args=default_args,
    schedule=None,          # manual trigger only -- no surprise scheduled runs
    start_date=datetime(2026, 1, 1),
    catchup=False,           # don't backfill runs for past dates
    tags=["churn", "mlops"],
) as dag:

    validate_task = BashOperator(
        task_id="validate",
        bash_command=f"cd {PROJECT_DIR} && python src/validate.py",
    )

    train_task = BashOperator(
        task_id="train",
        bash_command=f"cd {PROJECT_DIR} && python src/train.py",
        env={
            "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
            "AWS_ACCESS_KEY_ID": "minioadmin",
            "AWS_SECRET_ACCESS_KEY": "minioadmin",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        },

        
    )

    register_task = BashOperator(
        task_id="register_model",
        bash_command=f"cd {PROJECT_DIR} && python src/register_model.py",
        env={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "AWS_ACCESS_KEY_ID": "minioadmin",
            "AWS_SECRET_ACCESS_KEY": "minioadmin",
            "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
        },
    )

    validate_task >> train_task >> register_task
