"""
modelling.py (MLProject version)
=================================
MLflow Project entry point for CI/CD model training.
Designed to be invoked via `mlflow run` command.

This script loads preprocessed data, trains a RandomForestRegressor
with manual logging, and saves the model artifact.
"""

import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    max_error,
    explained_variance_score,
    median_absolute_error,
)

warnings.filterwarnings("ignore")


def load_data(data_dir: str):
    """Load preprocessed train/test CSVs."""
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    print(f"[load_data] Train: {train_df.shape} | Test: {test_df.shape}")
    return train_df, test_df


def train(
    data_dir: str,
    n_estimators: int = 200,
    max_depth: int = 15,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
):
    """Train model with manual MLflow logging."""

    train_df, test_df = load_data(data_dir)

    TARGET = "Price"
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]
    feature_names = list(X_train.columns)

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("random_state", 42)

        # Train
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        med_ae = median_absolute_error(y_test, y_pred)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("max_error", max_err)
        mlflow.log_metric("explained_variance_score", evs)
        mlflow.log_metric("median_absolute_error", med_ae)

        print(f"  MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | R²: {r2:.4f}")

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="Melbourne-Housing-RF-CI",
        )

        # Extra artifact: feature importance
        os.makedirs("artifacts", exist_ok=True)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(importances)), importances[indices], color="#4C72B0")
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        ax.invert_yaxis()
        plt.tight_layout()
        fi_path = "artifacts/feature_importance.png"
        plt.savefig(fi_path, dpi=100)
        plt.close()
        mlflow.log_artifact(fi_path, "plots")

        # Extra artifact: model summary
        summary = {
            "run_id": run.info.run_id,
            "metrics": {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2},
            "params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
            },
            "features": feature_names,
        }
        summary_path = "artifacts/model_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path, "metadata")

        # Tags
        mlflow.set_tag("model_type", "RandomForestRegressor")
        mlflow.set_tag("pipeline", "CI")
        mlflow.set_tag("author", "RizalMujahiddan")

        print(f"  Run ID: {run.info.run_id}")
        print(f"  Model and artifacts logged successfully.")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="Melbourne_housing_preprocessing")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
    )
