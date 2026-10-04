# SuperKart — Advanced Machine Learning and MLOps

**Project:** Automated Sales Forecasting Pipeline with CI/CD  
**Hugging Face username:** `dinzprof`  
**GitHub username:** `DineshrajR`  
**Repository:** `dinzprof`

## Business objective

Build a reproducible MLOps workflow that transforms SuperKart product/store data into reliable sales predictions, registers datasets and models on Hugging Face, and continuously deploys a Streamlit application through GitHub Actions.

## Rubric coverage

| Rubric area | Implementation |
|---|---|
| Data Registration | `data/raw/SuperKart.csv` + `src/data_prep.py` registers the dataset on Hugging Face |
| Data Preparation | HF-first ingestion, cleaning, feature engineering, 80/20 split, local persistence, HF upload |
| Model Training & Registration | 5 approved tree algorithms + tuned Random Forest; metrics and model registered |
| Model Deployment | Docker + Streamlit + HF Spaces hosting script |
| GitHub Actions | End-to-end CI/CD on `main` + manual dispatch + automated results commit |
| Output Evaluation | Live GitHub/HF links and notebook evidence section |
| Notebook Quality | Executed notebook, comments, observations, visuals, no intentional error cells |

## Live resources

- GitHub: https://github.com/DineshrajR/dinzprof
- Hugging Face dataset: https://huggingface.co/datasets/dinzprof/superkart-sales-data
- Hugging Face model: https://huggingface.co/dinzprof/superkart-sales-model
- Hugging Face Space: https://huggingface.co/spaces/dinzprof/superkart-sales-app

## Repository structure

```text
dinzprof/
├── .github/workflows/pipeline.yml
├── data/
│   ├── raw/SuperKart.csv
│   └── processed/
├── deployment/
│   ├── app.py
│   ├── Dockerfile
│   ├── README.md
│   ├── hosting.py
│   └── requirements.txt
├── notebooks/
│   ├── SuperKart_MLOps_Pipeline.ipynb
│   └── SuperKart_MLOps_Pipeline.html
├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── train.py
│   └── utils.py
├── docs/
├── requirements.txt
└── README.md
```

## GitHub secret

Add repository secret `HF_TOKEN` containing a Hugging Face **write** token. Never commit the token to source control.

## Local execution

```bash
pip install -r requirements.txt
python -m src.data_prep
python -m src.train
streamlit run deployment/app.py
```

## Reproducibility

- Train/test seed: `42`
- Test size: `20%`
- Store-age reference year: `2026`
- Same engineered feature schema in training and deployment.
- Best model is saved as `best_model.joblib` and registered on Hugging Face.
