"""
Hosting script: pushes all deployment files (Streamlit app, Dockerfile,
requirements) into a Hugging Face Space so the pipeline's model is served
as a live, public web app.

Run:
    python -m deployment.hosting
"""

import os
import sys

from src import config

SPACE_README = f"""---
title: SuperKart Sales Forecast
emoji: \U0001F6D2
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# SuperKart Sales Forecast

Streamlit app that predicts `Product_Store_Sales_Total` using the best
model trained in the SuperKart MLOps pipeline and registered at
[{config.MODEL_REPO_ID}](https://huggingface.co/{config.MODEL_REPO_ID}).
"""


def main():
    if not os.environ.get("HF_TOKEN"):
        print("[hosting] HF_TOKEN not set -> skipping Space deployment.")
        return False
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[hosting] huggingface_hub not installed -> skipping Space deployment.")
        return False

    api = HfApi(token=config.HF_TOKEN)
    api.create_repo(repo_id=config.SPACE_REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)

    readme_path = "deployment/README.md"
    with open(readme_path, "w") as f:
        f.write(SPACE_README)

    for local_path, repo_path in [
        (readme_path, "README.md"),
        ("deployment/app.py", "app.py"),
        ("deployment/Dockerfile", "Dockerfile"),
        ("deployment/requirements.txt", "requirements.txt"),
    ]:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=config.SPACE_REPO_ID,
            repo_type="space",
        )

    api.add_space_secret(repo_id=config.SPACE_REPO_ID, key="MODEL_REPO_ID", value=config.MODEL_REPO_ID)
    if config.HF_TOKEN:
        api.add_space_secret(repo_id=config.SPACE_REPO_ID, key="HF_TOKEN", value=config.HF_TOKEN)

    print(f"[hosting] Space deployed: https://huggingface.co/spaces/{config.SPACE_REPO_ID}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
