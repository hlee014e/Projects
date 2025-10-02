# Credit Risk Modeling (Notebook)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hlee014e/Projects/blob/main/Credit-Risk/Credit_risk.ipynb)

This project builds an explainable **credit risk model** that predicts the probability of default (PD) and shows **why** with SHAP.  
It includes data loading, preprocessing, class-imbalance handling, model training (Logistic Regression baseline + XGBoost), evaluation, and interpretability.

---

## Data

The dataset contains Weight of Evidence (WOE)-transformed features for credit risk modeling.

- **File:** [`features-woe(Credit-Risk).csv`](https://github.com/hlee014e/Projects/blob/main/Credit-Risk/features-woe(Credit-Risk).csv) *(stored in the repo root)*  
- **Target:** `target_bad` (1 = default, 0 = non-default)  
- **Features:** WOE-transformed versions of key credit attributes (e.g., `int_rate`, `dti`, `annual_inc`, revolving utilization, balances, account counts, etc.)

> If you open this notebook in Colab, it will automatically download the CSV from GitHub.  
> If you run locally, ensure that the file is present in the same folder as the notebook, or update the file path.

---

## What’s inside

- **Notebook:** `Credit-Risk/Credit_risk.ipynb`
- **Goals:** Estimate borrower probability of default (PD)  
- **Methods:** Logistic Regression (baseline), XGBoost with imbalance handling, SHAP interpretability  
- **Metrics:** ROC-AUC, PR-AUC, KS statistic, confusion matrix, threshold analysis, calibration metrics  
- **Artifacts:** Feature importance, SHAP global summary, SHAP local (row-level) explanations
