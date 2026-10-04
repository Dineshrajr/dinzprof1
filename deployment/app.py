import os,joblib,pandas as pd,streamlit as st
from huggingface_hub import hf_hub_download
MODEL_REPO_ID=os.getenv("MODEL_REPO_ID","dinzprof/superkart-sales-model"); HF_TOKEN=os.getenv("HF_TOKEN"); YEAR=int(os.getenv("REFERENCE_YEAR","2026"))
TYPES=["Frozen Foods","Dairy","Canned","Baking Goods","Health and Hygiene","Snack Foods","Meat","Household","Hard Drinks","Fruits and Vegetables","Breads","Soft Drinks","Breakfast","Others","Starchy Foods","Seafood"]
@st.cache_resource
def model(): return joblib.load(hf_hub_download(repo_id=MODEL_REPO_ID,filename="best_model.joblib",repo_type="model",token=HF_TOKEN))
st.set_page_config(page_title="SuperKart Sales Forecast",page_icon="🛒",layout="wide"); st.title("🛒 SuperKart Sales Forecast"); st.caption("CI/CD deployed sales prediction application.")
with st.form("form"):
    a,b=st.columns(2)
    with a:
        w=st.number_input("Product Weight",0.0,30.0,12.5); sugar=st.selectbox("Product Sugar Content",["Low Sugar","Regular","No Sugar"]); area=st.number_input("Product Allocated Area",.001,1.,.05,format="%.3f"); typ=st.selectbox("Product Type",TYPES); mrp=st.number_input("Product MRP",0.,500.,145.)
    with b:
        year=st.number_input("Store Establishment Year",1980,YEAR,2009); size=st.selectbox("Store Size",["Small","Medium","High"]); city=st.selectbox("Store Location City Type",["Tier 1","Tier 2","Tier 3"]); store=st.selectbox("Store Type",["Food Mart","Supermarket Type1","Supermarket Type2","Departmental Store"])
    go=st.form_submit_button("Predict Sales",type="primary")
if go:
    X=pd.DataFrame([{"Product_Weight":w,"Product_Sugar_Content":sugar,"Product_Allocated_Area":area,"Product_Type":typ,"Product_MRP":mrp,"Store_Size":size,"Store_Location_City_Type":city,"Store_Type":store,"Store_Age":YEAR-int(year)}])
    try: st.success(f"### Predicted Total Sales: ₹{float(model().predict(X)[0]):,.2f}"); st.dataframe(X,use_container_width=True)
    except Exception as e: st.error(f"Prediction failed: {e}")
