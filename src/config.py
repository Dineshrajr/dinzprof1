from pathlib import Path
import os
PROJECT_ROOT=Path(__file__).resolve().parents[1]
RAW_DATA_PATH=PROJECT_ROOT/"data"/"raw"/"SuperKart.csv"
PROCESSED_DIR=PROJECT_ROOT/"data"/"processed"
TRAIN_PATH=PROCESSED_DIR/"train.csv"; TEST_PATH=PROCESSED_DIR/"test.csv"
BEST_MODEL_PATH=PROCESSED_DIR/"best_model.joblib"; METRICS_PATH=PROCESSED_DIR/"metrics.json"
RANDOM_STATE=42; TEST_SIZE=0.20; REFERENCE_YEAR=2026
TARGET_COLUMN="Product_Store_Sales_Total"
HF_USERNAME=os.getenv("HF_USERNAME","dinzprof")
DATASET_REPO_ID=f"{HF_USERNAME}/superkart-sales-data"
MODEL_REPO_ID=f"{HF_USERNAME}/superkart-sales-model"
SPACE_REPO_ID=f"{HF_USERNAME}/superkart-sales-app"
HF_TOKEN=os.getenv("HF_TOKEN")
GITHUB_REPO_URL="https://github.com/DineshrajR/dinzprof"
HF_DATASET_URL=f"https://huggingface.co/datasets/{DATASET_REPO_ID}"
HF_MODEL_URL=f"https://huggingface.co/{MODEL_REPO_ID}"
HF_SPACE_URL=f"https://huggingface.co/spaces/{SPACE_REPO_ID}"
NUMERIC_FEATURES=["Product_Weight","Product_Allocated_Area","Product_MRP","Store_Age"]
CATEGORICAL_FEATURES=["Product_Sugar_Content","Product_Type","Store_Size","Store_Location_City_Type","Store_Type"]
MODEL_FEATURES=NUMERIC_FEATURES+CATEGORICAL_FEATURES
DROP_COLUMNS=["Product_Id","Store_Id","Store_Establishment_Year"]
