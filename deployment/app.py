"""
Streamlit front-end for the SuperKart sales forecast model.

Loads the best model registered on the Hugging Face model hub, collects
inputs from the user into a single-row dataframe (matching the schema the
model was trained on), and returns the predicted total sales for that
product/store combination.

Deployed to a Hugging Face Space (SDK: docker) built from ../deployment/Dockerfile.
"""

import datetime
import os

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "your-hf-username/superkart-sales-model")
HF_TOKEN = os.environ.get("HF_TOKEN")  # optional: only needed for private repos


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="best_model.joblib",
        repo_type="model",
        token=HF_TOKEN,
    )
    return joblib.load(model_path)


def main():
    st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒")
    st.title("🛒 SuperKart Sales Forecast")
    st.write(
        "Predict the total sales revenue for a product at a given store, "
        "using the model trained and registered in the SuperKart MLOps pipeline."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            product_weight = st.number_input("Product Weight", min_value=0.0, value=12.5)
            product_sugar_content = st.selectbox(
                "Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"]
            )
            product_allocated_area = st.slider(
                "Product Allocated Area (ratio of store display area)",
                min_value=0.0, max_value=1.0, value=0.05,
            )
            product_type = st.selectbox(
                "Product Type",
                [
                    "Frozen Foods", "Dairy", "Canned", "Baking Goods",
                    "Health and Hygiene", "Snack Foods", "Meat", "Household",
                    "Hard Drinks", "Fruits and Vegetables", "Breads",
                    "Soft Drinks", "Breakfast", "Others", "Starchy Foods", "Seafood",
                ],
            )
            product_mrp = st.number_input("Product MRP", min_value=0.0, value=140.0)

        with col2:
            store_establishment_year = st.number_input(
                "Store Establishment Year", min_value=1950, max_value=datetime.datetime.now().year,
                value=1999, step=1,
            )
            store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
            store_location_city_type = st.selectbox(
                "Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]
            )
            store_type = st.selectbox(
                "Store Type",
                ["Food Mart", "Supermarket Type1", "Supermarket Type2", "Departmental Store"],
            )

        submitted = st.form_submit_button("Predict Sales")

    if submitted:
        store_age = datetime.datetime.now().year - int(store_establishment_year)
        input_df = pd.DataFrame(
            [
                {
                    "Product_Weight": product_weight,
                    "Product_Sugar_Content": product_sugar_content,
                    "Product_Allocated_Area": product_allocated_area,
                    "Product_Type": product_type,
                    "Product_MRP": product_mrp,
                    "Store_Establishment_Year": store_establishment_year,
                    "Store_Size": store_size,
                    "Store_Location_City_Type": store_location_city_type,
                    "Store_Type": store_type,
                    "Store_Age": store_age,
                }
            ]
        )

        model = load_model()
        prediction = model.predict(input_df)[0]
        st.success(f"### Predicted Total Sales: ₹{prediction:,.2f}")
        st.caption("Model: best pipeline registered on the Hugging Face model hub.")
        with st.expander("Input sent to the model"):
            st.dataframe(input_df)


if __name__ == "__main__":
    main()
