"""
Data registration + preparation step of the SuperKart MLOps pipeline.

Rubric coverage:
  - Load the dataset directly from the Hugging Face dataset space
    (falls back to the local raw CSV the first time, before the raw
    file itself has been registered on the Hub).
  - Clean the data and remove unnecessary columns.
  - Split into train/test sets and save them locally.
  - Upload the resulting train/test datasets back to the Hugging Face
    dataset space.

Run:
    python -m src.data_prep
"""

import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config, utils


def register_raw_data():
    """Push the raw CSV to the HF dataset repo (one-time registration)."""
    if not utils.hf_ready():
        print("[data_prep] HF not configured -> skipping raw data registration.")
        return
    api = utils.get_hf_api()
    api.create_repo(
        repo_id=config.DATASET_REPO_ID, repo_type="dataset", exist_ok=True
    )
    api.upload_file(
        path_or_fileobj=config.RAW_DATA_PATH,
        path_in_repo="raw/SuperKart.csv",
        repo_id=config.DATASET_REPO_ID,
        repo_type="dataset",
    )
    print(f"[data_prep] Raw data registered at {config.DATASET_REPO_ID}")


def load_raw_data():
    """Load the raw dataset, preferring the Hugging Face dataset space."""
    if utils.hf_ready():
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=config.DATASET_REPO_ID,
                filename="raw/SuperKart.csv",
                repo_type="dataset",
                token=config.HF_TOKEN,
            )
            print(f"[data_prep] Loaded raw data from HF Hub: {config.DATASET_REPO_ID}")
            return pd.read_csv(path)
        except Exception as e:
            print(f"[data_prep] Could not load from HF Hub ({e}); using local file.")
    print(f"[data_prep] Loaded raw data from local file: {config.RAW_DATA_PATH}")
    return pd.read_csv(config.RAW_DATA_PATH)


def clean_data(df):
    """Clean the raw dataframe and drop unnecessary columns."""
    df = df.copy()

    before = df.shape
    df = df.drop_duplicates()

    # Normalise inconsistent category labels.
    df["Product_Sugar_Content"] = utils.clean_sugar_content(df["Product_Sugar_Content"])

    # Product_Id is a unique identifier per row (not predictive) and
    # Store_Id is a 1:1 stand-in for the store attributes we already keep
    # (Store_Size / Store_Location_City_Type / Store_Type / establishment
    # year), so both are dropped as unnecessary columns for modeling.
    df = df.drop(columns=config.ID_COLS)

    df = utils.engineer_features(df)

    print(f"[data_prep] Cleaned data: {before} -> {df.shape}")
    return df


def split_and_save(df):
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=config.RANDOM_STATE
    )
    utils.ensure_dir(config.TRAIN_DATA_PATH)
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)
    print(
        f"[data_prep] Saved train ({train_df.shape}) -> {config.TRAIN_DATA_PATH}, "
        f"test ({test_df.shape}) -> {config.TEST_DATA_PATH}"
    )
    return train_df, test_df


def upload_processed_data():
    if not utils.hf_ready():
        print("[data_prep] HF not configured -> skipping train/test upload.")
        return
    api = utils.get_hf_api()
    api.create_repo(
        repo_id=config.DATASET_REPO_ID, repo_type="dataset", exist_ok=True
    )
    for local_path, repo_path in [
        (config.TRAIN_DATA_PATH, "processed/train.csv"),
        (config.TEST_DATA_PATH, "processed/test.csv"),
    ]:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=config.DATASET_REPO_ID,
            repo_type="dataset",
        )
    print(f"[data_prep] train/test uploaded to {config.DATASET_REPO_ID}")


def main():
    register_raw_data()
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    train_df, test_df = split_and_save(clean_df)
    upload_processed_data()

    summary = {
        "raw_rows": int(raw_df.shape[0]),
        "clean_rows": int(clean_df.shape[0]),
        "train_rows": int(train_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
    }
    print("[data_prep] Summary:", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
