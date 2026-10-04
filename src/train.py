"""
Model training, tuning, evaluation, and registration step of the
SuperKart MLOps pipeline.

Rubric coverage:
  - Load the train/test data from the Hugging Face dataset space.
  - Define a model and parameters.
  - Tune the model with the defined parameters.
  - Evaluate model performance.
  - Register the best model in the Hugging Face model hub.

Algorithms compared: Decision Tree, Bagging, Random Forest, AdaBoost,
Gradient Boosting, and XGBoost (skipped automatically if the xgboost
package is not installed in the current environment).

Run:
    python -m src.train
"""

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src import config, utils

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def load_train_test():
    if utils.hf_ready():
        try:
            from huggingface_hub import hf_hub_download

            train_path = hf_hub_download(
                repo_id=config.DATASET_REPO_ID,
                filename="processed/train.csv",
                repo_type="dataset",
                token=config.HF_TOKEN,
            )
            test_path = hf_hub_download(
                repo_id=config.DATASET_REPO_ID,
                filename="processed/test.csv",
                repo_type="dataset",
                token=config.HF_TOKEN,
            )
            print(f"[train] Loaded train/test from HF Hub: {config.DATASET_REPO_ID}")
            return pd.read_csv(train_path), pd.read_csv(test_path)
        except Exception as e:
            print(f"[train] Could not load from HF Hub ({e}); using local files.")
    print("[train] Loaded train/test from local data/processed/*.csv")
    return pd.read_csv(config.TRAIN_DATA_PATH), pd.read_csv(config.TEST_DATA_PATH)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_COLS),
        ],
        remainder="passthrough",
    )


def get_model_grid():
    """Model + hyperparameter grid definitions.

    Kept intentionally small so the whole comparison finishes quickly in
    CI; widen these grids for a more exhaustive search.
    """
    grid = {
        "DecisionTree": (
            DecisionTreeRegressor(random_state=config.RANDOM_STATE),
            {"model__max_depth": [4, 6, 8, None], "model__min_samples_leaf": [1, 5, 10]},
        ),
        "Bagging": (
            BaggingRegressor(random_state=config.RANDOM_STATE),
            {"model__n_estimators": [50, 100], "model__max_samples": [0.7, 1.0]},
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=config.RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [6, 10, None],
                "model__min_samples_leaf": [1, 5],
            },
        ),
        "AdaBoost": (
            AdaBoostRegressor(random_state=config.RANDOM_STATE),
            {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1, 1.0]},
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=config.RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            },
        ),
    }
    if HAS_XGBOOST:
        grid["XGBoost"] = (
            XGBRegressor(
                random_state=config.RANDOM_STATE, objective="reg:squarederror"
            ),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 4, 6],
            },
        )
    else:
        print("[train] xgboost not installed in this environment -> skipping XGBoost.")
    return grid


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_and_tune(train_df, test_df):
    X_train = train_df.drop(columns=[config.TARGET_COL])
    y_train = train_df[config.TARGET_COL]
    X_test = test_df.drop(columns=[config.TARGET_COL])
    y_test = test_df[config.TARGET_COL]

    preprocessor = build_preprocessor()
    results = {}
    best_name, best_estimator, best_rmse = None, None, np.inf

    for name, (estimator, param_grid) in get_model_grid().items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        search = GridSearchCV(
            pipe, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        search.fit(X_train, y_train)
        metrics = evaluate(search.best_estimator_, X_test, y_test)
        metrics["best_params"] = search.best_params_
        results[name] = metrics
        print(f"[train] {name}: RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.4f}  "
              f"params={search.best_params_}")

        if metrics["rmse"] < best_rmse:
            best_name, best_estimator, best_rmse = name, search.best_estimator_, metrics["rmse"]

    print(f"[train] Best model: {best_name} (RMSE={best_rmse:.2f})")
    return best_name, best_estimator, results


def save_model_and_metrics(best_name, best_estimator, results):
    utils.ensure_dir(config.MODEL_LOCAL_PATH)
    joblib.dump(best_estimator, config.MODEL_LOCAL_PATH)
    with open(config.METRICS_LOCAL_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print(f"[train] Saved model -> {config.MODEL_LOCAL_PATH}")
    print(f"[train] Saved metrics -> {config.METRICS_LOCAL_PATH}")


def register_model():
    if not utils.hf_ready():
        print("[train] HF not configured -> skipping model registration.")
        return
    api = utils.get_hf_api()
    api.create_repo(repo_id=config.MODEL_REPO_ID, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=config.MODEL_LOCAL_PATH,
        path_in_repo="best_model.joblib",
        repo_id=config.MODEL_REPO_ID,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=config.METRICS_LOCAL_PATH,
        path_in_repo="metrics.json",
        repo_id=config.MODEL_REPO_ID,
        repo_type="model",
    )
    print(f"[train] Model registered at {config.MODEL_REPO_ID}")


def main():
    train_df, test_df = load_train_test()
    best_name, best_estimator, results = train_and_tune(train_df, test_df)
    save_model_and_metrics(best_name, best_estimator, results)
    register_model()
    return {"best_model": best_name, "results": results}


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
