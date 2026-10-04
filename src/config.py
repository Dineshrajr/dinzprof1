"""
Shared configuration for the SuperKart MLOps pipeline.

Hugging Face repo names are derived from the HF_USERNAME environment
variable so the same code works for any student/user without editing
source files. Set HF_USERNAME and HF_TOKEN as environment variables
(locally) or as GitHub Actions secrets (in CI).
"""

import os

# ---- Hugging Face identity ----
HF_USERNAME = os.environ.get("HF_USERNAME", "your-hf-username")
HF_TOKEN = os.environ.get("HF_TOKEN")  # never hard-code this

# ---- Hugging Face repo ids ----
DATASET_REPO_ID = f"{HF_USERNAME}/superkart-sales-data"
MODEL_REPO_ID = f"{HF_USERNAME}/superkart-sales-model"
SPACE_REPO_ID = f"{HF_USERNAME}/superkart-sales-app"

# ---- Local paths (relative to project root) ----
RAW_DATA_PATH = "data/raw/SuperKart.csv"
TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"
MODEL_LOCAL_PATH = "data/processed/best_model.joblib"
METRICS_LOCAL_PATH = "data/processed/metrics.json"

# ---- Columns ----
TARGET_COL = "Product_Store_Sales_Total"
ID_COLS = ["Product_Id", "Store_Id"]  # identifiers, not predictive features

CATEGORICAL_COLS = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
]

NUMERIC_COLS = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year",
    "Store_Age",  # engineered
]

RANDOM_STATE = 42
