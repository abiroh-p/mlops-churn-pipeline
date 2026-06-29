"""
Pass 2: Train a churn classifier with a sklearn Pipeline, tracked in MLflow.
"""

import mlflow
import mlflow.sklearn
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA_PATH = "data/processed/telco_churn_clean.csv"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "churn-prediction"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_pipeline(numeric_cols, categorical_cols, max_iter: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=max_iter)),
        ]
    )
    return pipeline


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    test_size = 0.2
    random_state = 42
    max_iter = 1000

    df = load_data(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    with mlflow.start_run():
        # --- log inputs we chose ---
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_numeric_features", len(numeric_cols))
        mlflow.log_param("n_categorical_features", len(categorical_cols))

        pipeline = build_pipeline(numeric_cols, categorical_cols, max_iter)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        # --- log outputs we measured ---
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # --- log the fitted pipeline itself as an artifact ---
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC:  {auc:.4f}")
        print(f"\nLogged to MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()