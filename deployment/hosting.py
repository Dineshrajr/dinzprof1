from pathlib import Path
from src import config
def main():
    if not config.HF_TOKEN: raise RuntimeError("HF_TOKEN required.")
    from huggingface_hub import HfApi
    a=HfApi(token=config.HF_TOKEN); a.create_repo(repo_id=config.SPACE_REPO_ID,repo_type="space",space_sdk="docker",exist_ok=True)
    for f in ["app.py","Dockerfile","requirements.txt","README.md"]:
        a.upload_file(path_or_fileobj=str(Path(__file__).with_name(f)),path_in_repo=f,repo_id=config.SPACE_REPO_ID,repo_type="space",commit_message="Deploy "+f)
if __name__=="__main__": main()
