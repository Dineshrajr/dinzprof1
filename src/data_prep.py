import pandas as pd
from sklearn.model_selection import train_test_split
from . import config
from .utils import clean_superkart,ensure_dirs
SOURCE_REPO="dev02chandan/sales-forecast-dataset"; SOURCE_FILE="raw/SuperKart.csv"
def api():
    if not config.HF_TOKEN: return None
    from huggingface_hub import HfApi
    return HfApi(token=config.HF_TOKEN)
def download_source():
    ensure_dirs()
    if config.RAW_DATA_PATH.exists(): return
    from huggingface_hub import hf_hub_download
    p=hf_hub_download(repo_id=SOURCE_REPO,filename=SOURCE_FILE,repo_type="dataset")
    pd.read_csv(p).to_csv(config.RAW_DATA_PATH,index=False)
def load_raw_data():
    if config.HF_TOKEN:
        try:
            from huggingface_hub import hf_hub_download
            p=hf_hub_download(repo_id=config.DATASET_REPO_ID,filename="raw/SuperKart.csv",repo_type="dataset",token=config.HF_TOKEN)
            return pd.read_csv(p)
        except Exception as exc: print("HF-first load not ready:",type(exc).__name__)
    download_source(); return pd.read_csv(config.RAW_DATA_PATH)
def prepare_and_split():
    ensure_dirs(); cleaned=clean_superkart(load_raw_data())
    tr,te=train_test_split(cleaned,test_size=config.TEST_SIZE,random_state=config.RANDOM_STATE)
    tr=tr.reset_index(drop=True); te=te.reset_index(drop=True)
    tr.to_csv(config.TRAIN_PATH,index=False); te.to_csv(config.TEST_PATH,index=False); return tr,te
def register_raw_data():
    a=api()
    if not a: return False
    a.create_repo(repo_id=config.DATASET_REPO_ID,repo_type="dataset",exist_ok=True)
    a.upload_file(path_or_fileobj=str(config.RAW_DATA_PATH),path_in_repo="raw/SuperKart.csv",repo_id=config.DATASET_REPO_ID,repo_type="dataset",commit_message="Register raw SuperKart dataset"); return True
def upload_processed_data():
    a=api()
    if not a: return False
    for p,r in [(config.TRAIN_PATH,"processed/train.csv"),(config.TEST_PATH,"processed/test.csv")]:
        a.upload_file(path_or_fileobj=str(p),path_in_repo=r,repo_id=config.DATASET_REPO_ID,repo_type="dataset",commit_message="Upload "+r)
    return True
if __name__=="__main__":
    prepare_and_split(); register_raw_data(); upload_processed_data()
