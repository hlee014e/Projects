# Credit Risk Modeling (Notebook)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hlee014e/Projects/blob/main/Credit-Risk/Credit_risk.ipynb)

This project builds an explainable **credit risk model** that predicts the probability of default (PD) and shows **why** with SHAP.  
It includes data loading, preprocessing, class-imbalance handling, model training (Logistic Regression baseline + XGBoost), evaluation, and interpretability.

---

## Data

- **File:** `features-woe(Credit-Risk).csv` *(stored in the repo root)*  
- **Target:** `target_bad` (1 = default, 0 = non-default)  
- Features include WOE-transformed variables (e.g., `int_rate`, `dti`, `annual_inc`, credit utilization and balances, account counts, etc.)

> If you open in Colab, the notebook will download this CSV directly from GitHub.  
> If you run locally, ensure the CSV is present in the same folder as the notebook or update the path.

**Quick load snippet (works in Colab & locally):**
```python
import pandas as pd, io, requests, os

CSV_FILE = "features-woe(Credit-Risk).csv"
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    # fallback: load from GitHub raw (replace with your exact repo path if needed)
    url = "https://raw.githubusercontent.com/hlee014e/Projects/main/features-woe(Credit-Risk).csv"
    df = pd.read_csv(url)

df.head()
