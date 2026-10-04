import joblib,numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from . import config
from .data_prep import prepare_and_split
from .utils import save_json
def preprocessor():
    return ColumnTransformer([("num",SimpleImputer(strategy="median"),config.NUMERIC_FEATURES),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),config.CATEGORICAL_FEATURES)])
def pipe(model): return Pipeline([("preprocess",preprocessor()),("model",model)])
def score(model,X,y):
    p=model.predict(X); return {"rmse":float(np.sqrt(mean_squared_error(y,p))),"mae":float(mean_absolute_error(y,p)),"r2":float(r2_score(y,p))}
def train_and_tune():
    tr,te=prepare_and_split(); Xtr=tr[config.MODEL_FEATURES]; ytr=tr[config.TARGET_COLUMN]; Xte=te[config.MODEL_FEATURES]; yte=te[config.TARGET_COLUMN]
    candidates={"DecisionTree":DecisionTreeRegressor(min_samples_leaf=10,random_state=42),"Bagging":BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42),n_estimators=100,max_samples=.7,random_state=42,n_jobs=-1),"RandomForest":RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=42,n_jobs=-1),"AdaBoost":AdaBoostRegressor(n_estimators=100,learning_rate=.05,random_state=42),"GradientBoosting":GradientBoostingRegressor(n_estimators=200,learning_rate=.1,max_depth=4,random_state=42)}
    results={}; fitted={}
    for n,m in candidates.items(): fitted[n]=pipe(m).fit(Xtr,ytr); results[n]=score(fitted[n],Xte,yte)
    gs=GridSearchCV(pipe(RandomForestRegressor(random_state=42,n_jobs=-1)),{"model__n_estimators":[100,200],"model__max_depth":[None,20],"model__min_samples_leaf":[1,5]},cv=3,scoring="neg_root_mean_squared_error",n_jobs=-1).fit(Xtr,ytr)
    results["RandomForest_Tuned"]=score(gs.best_estimator_,Xte,yte); best_name=min(results,key=lambda n:results[n]["rmse"]); best=gs.best_estimator_ if best_name=="RandomForest_Tuned" else fitted[best_name]
    config.PROCESSED_DIR.mkdir(parents=True,exist_ok=True); joblib.dump(best,config.BEST_MODEL_PATH)
    save_json({"best_model":best_name,"best_rmse":results[best_name]["rmse"],"best_mae":results[best_name]["mae"],"best_r2":results[best_name]["r2"],"results":results,"features":config.MODEL_FEATURES},config.METRICS_PATH)
    print("Best model:",best_name,"RMSE:",round(results[best_name]["rmse"],2),"R2:",round(results[best_name]["r2"],4)); return best
def register_model():
    if not config.HF_TOKEN: raise RuntimeError("HF_TOKEN required.")
    from huggingface_hub import HfApi
    a=HfApi(token=config.HF_TOKEN); a.create_repo(repo_id=config.MODEL_REPO_ID,repo_type="model",exist_ok=True)
    for p,r in [(config.BEST_MODEL_PATH,"best_model.joblib"),(config.METRICS_PATH,"metrics.json")]:
        a.upload_file(path_or_fileobj=str(p),path_in_repo=r,repo_id=config.MODEL_REPO_ID,repo_type="model",commit_message="Register "+r)
if __name__=="__main__": train_and_tune(); register_model()
