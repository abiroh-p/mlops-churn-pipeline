"""
Registers the most recent training run's model into MLflow's Model Registry,
and promotes it to Production if it beats the current Production model on F1.

Run this after train.py. In the DAG, this becomes a third task:
validate -> train -> register_model
"""

import os
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "churn-prediction"
REGISTERED_MODEL_NAME = "churn-classifier"


def get_latest_run(client: MlflowClient):
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No runs found in experiment '{EXPERIMENT_NAME}'")
    return runs[0]


def get_production_f1(client: MlflowClient) -> float | None:
    """Returns F1 of the current Production model, or None if nothing is promoted yet."""
    try:
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    except mlflow.exceptions.MlflowException:
        return None  # registered model doesn't exist yet

    if not versions:
        return None

    prod_version = versions[0]
    prod_run = client.get_run(prod_version.run_id)
    return prod_run.data.metrics.get("f1_score")


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    latest_run = get_latest_run(client)
    latest_f1 = latest_run.data.metrics.get("f1_score")
    run_id = latest_run.info.run_id

    print(f"Latest run: {run_id}")
    print(f"Latest run F1: {latest_f1:.4f}")

    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    new_version = registered_model.version

    print(f"Registered as {REGISTERED_MODEL_NAME} version {new_version}")

    current_prod_f1 = get_production_f1(client)

    if current_prod_f1 is None:
        print("No current Production model -- promoting this version.")
        should_promote = True
    elif latest_f1 > current_prod_f1:
        print(f"New F1 ({latest_f1:.4f}) beats current Production F1 ({current_prod_f1:.4f}) -- promoting.")
        should_promote = True
    else:
        print(f"New F1 ({latest_f1:.4f}) does not beat current Production F1 ({current_prod_f1:.4f}) -- not promoting.")
        should_promote = False

    if should_promote:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=new_version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Version {new_version} is now in Production stage.")
    else:
        print(f"Version {new_version} registered but left in stage: None")


if __name__ == "__main__":
    main()