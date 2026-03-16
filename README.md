# Sovereign Debt Crisis Forecasting

Machine learning pipeline for predicting sovereign debt crises using macroeconomic, financial, and institutional indicators across 60+ countries (1970–2021).

## Overview

This project implements and compares multiple forecasting approaches:

- **Base models** — Rule of Thumb, Probit, Random Forest, AdaBoost, XGBoost
- **Sequence-RF meta-learners** — LSTM and PatchTST generate macro forecast features fed into a Random Forest
- **Stacked meta-learner** — GRU-based stacked ensemble over base model predictions
- **FAVAR-Net meta-learner** — Factor-Gated Network (Mixture-of-Experts + Gated Residual Network) with engineered features and SHAP-based feature selection

Evaluation uses rolling-origin cross-validation with DeLong tests (AUC), paired bootstrap tests (MSE, log-likelihood, F1), and SHAP-based interpretability analysis.

## Project Structure

```
data/                          Data construction and train/test splitting
  construction.py              Build baseline (20-feature) and expanded (42-feature) datasets
  make_splits.py               Horizon-specific splits with imputation and winsorization

metrics/                       Shared evaluation and significance testing
  evaluation.py                AUC, F1, MSE, log-likelihood, confusion matrix
  significance_tests.py        DeLong test, paired bootstrap

basemodels/                    Baseline classifiers (ROT, Probit, RF, AdaBoost, XGBoost)
shap_selection/                SHAP feature importance ranking (TreeExplainer)
shap_model/                    RF trained on SHAP-selected feature subsets
superlearners/                 LSTM + PatchTST sequence-RF meta-learners
superlearner_stacked/          GRU stacked ensemble meta-learner
superlearner_favar_shuffle/    FAVAR-Net meta-learner (MoE + GRN architecture)
```

## Usage

Each module is run as a Python package from the project root. Example:

```bash
# 1. Build datasets from raw sources
python -m data.construction

# 2. Create train/test splits for each horizon
python -m data.make_splits

# 3. Train base models
python -m basemodels.run_basemodels --dataset baseline --horizon 2 --group ALL
python -m basemodels.run_basemodels --dataset expanded --horizon 2 --group ALL

# 4. Compare baseline vs expanded
python -m basemodels.evaluate_baseline_vs_expanded --horizon 2 --group ALL

# 5. SHAP feature selection
python -m shap_selection.run_shap --dataset expanded --horizon 2 --group ALL

# 6. SHAP-selected models
python -m shap_model.run_shap_model --horizon 2 --group ALL

# 7. Sequence-RF meta-learners
python -m superlearners.run_superlearners --dataset expanded --horizon 2 --group ALL

# 8. Stacked meta-learner
python -m superlearner_stacked.run_stacked --dataset expanded --horizon 2 --group ALL

# 9. FAVAR-Net meta-learner
python -m superlearner_favar_shuffle.run_superlearners --dataset expanded --horizon 2 --group ALL
```

## Data

Raw datasets must be sourced independently and placed in `data/raw_datasets/`. Required sources include:

- **World Development Indicators (WDI)** — World Bank
- **World Economic Outlook (WEO)** — IMF
- **Global Macro Database (GMD)**
- **Quality of Government (QoG)** — University of Gothenburg
- **Worldwide Governance Indicators (WGI)** — World Bank
- **Capital Account Openness (ka_open)** — Chinn-Ito Index
- **Global macro series** — VIX, commodity prices, oil prices, interest rates

See `data/construction.py` for the full list of required files and indicator codes.

## Requirements

- Python 3.9+
- numpy, pandas, scikit-learn, statsmodels, xgboost, shap
- tensorflow/keras (for LSTM, PatchTST, GRU, FAVAR-Net models)
- scipy, joblib, openpyxl

## Forecast Horizons

| Horizon | Label | Test Windows |
|---------|-------|-------------|
| h = 2   | Medium-term | 2000–2019 |
| h = 5   | Long-term | 2000–2016 |
| h = 10  | Long-term | 2000–2011 |

## License

This project is provided for academic and research purposes.
