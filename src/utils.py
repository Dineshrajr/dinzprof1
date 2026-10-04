"""Shared helper functions for the SuperKart MLOps pipeline."""

import datetime
import os

from src import config


def hf_ready():
    """Return True if huggingface_hub is installed AND a token is set.

    This lets every script run in two modes:
      - Local/offline mode (e.g. this sandbox, or a laptop with no token):
        all local steps (clean, split, train, evaluate) still run and are
        verified; HF push steps are skipped with a clear message.
      - Full mode (GitHub Actions, or a machine with HF_TOKEN exported):
        data/model/Space are actually pushed to the Hugging Face Hub.
    """
    if not config.HF_TOKEN:
        return False
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return False
    return True


def get_hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=config.HF_TOKEN)


def clean_sugar_content(series):
    """Normalise inconsistent labels in Product_Sugar_Content."""
    mapping = {
        "reg": "Regular",
        "Regular": "Regular",
        "Low Sugar": "Low Sugar",
        "No Sugar": "No Sugar",
    }
    return series.replace(mapping)


def engineer_features(df):
    """Add engineered features used by the model."""
    df = df.copy()
    current_year = datetime.datetime.now().year
    df["Store_Age"] = current_year - df["Store_Establishment_Year"]
    return df


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
