import json
from pathlib import Path
import pandas as pd
from . import config
def ensure_dirs():
    config.PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
    (config.PROJECT_ROOT/"data"/"raw").mkdir(parents=True,exist_ok=True)
def engineer_features(df):
    out=df.copy()
    if "Product_Sugar_Content" in out: out["Product_Sugar_Content"]=out["Product_Sugar_Content"].replace({"reg":"Regular"})
    if "Store_Establishment_Year" in out: out["Store_Age"]=config.REFERENCE_YEAR-out["Store_Establishment_Year"]
    return out
def clean_superkart(df):
    out=engineer_features(df).drop(columns=[c for c in config.DROP_COLUMNS if c in df.columns]).drop_duplicates().reset_index(drop=True)
    out=out[[c for c in config.MODEL_FEATURES+[config.TARGET_COLUMN] if c in out.columns]]
    for c in config.NUMERIC_FEATURES: out[c]=pd.to_numeric(out[c],errors="coerce")
    for c in config.CATEGORICAL_FEATURES: out[c]=out[c].astype("string").str.strip()
    if out[config.MODEL_FEATURES].isna().any().any(): raise ValueError("Missing feature values remain after cleaning.")
    return out
def save_json(payload,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")
