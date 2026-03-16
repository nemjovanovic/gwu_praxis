# ================================================================
# Base Models Configuration
# ================================================================
#
# Hyperparameters for all base models.
#
# ================================================================

import os

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_BASE = os.path.join(PROJECT_ROOT, "data", "final")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "basemodels", "results")

# ----------------------------------------------------------------
# Dataset / Split Configuration
# ----------------------------------------------------------------
VALID_DATASETS = ["baseline", "expanded"]
VALID_HORIZONS = [2, 5, 10]
VALID_GROUPS = ["ALL", "EME", "LIC"]

# Rolling window parameters
LOOKBACK = 10
T_MIN = 2000

# T_MAX depends on horizon: 2021 - h
def get_t_max(horizon: int) -> int:
    return 2021 - horizon

# ----------------------------------------------------------------
# Feature Columns (exclude non-feature columns)
# ----------------------------------------------------------------
NON_FEATURE_COLS = [
    "WEOCountryCode",
    "GroupName",
    "Year",
    "crisis",
    "c",   # Current year crisis indicator
    "CH",  # Crisis history (used by Rule of Thumb, not as feature)
]

TARGET_COL = "crisis"
COUNTRY_COL = "WEOCountryCode"
YEAR_COL = "Year"
CH_COL = "CH"  # Crisis history column for Rule of Thumb

# ----------------------------------------------------------------
# Cross-Validation Configuration
# ----------------------------------------------------------------
N_FOLDS = 10  # Country-based CV

# ----------------------------------------------------------------
# Rule of Thumb - No hyperparameters
# Formula: 1 - ((1 - CH/100)^h)
# ----------------------------------------------------------------
ROT_CONFIG = {
    # No tunable parameters
}

# ----------------------------------------------------------------
# Probit Regression
# GLM with probit link function
# ----------------------------------------------------------------
PROBIT_CONFIG = {
    "max_iter": 1000,
    "random_state": 1234,
}

# ----------------------------------------------------------------
# Random Forest Hyperparameters
# ----------------------------------------------------------------
RF_CONFIG = {
    "n_estimators": 1000,
    "mtry_grid": [3, 4, 5, 6, 7, 8, 9, 10],
    "cv_folds": 10,
    "random_state": 1234,
    "n_jobs": -1,
    "class_weight": "balanced",
}

# ----------------------------------------------------------------
# AdaBoost (Gradient Boosting) Hyperparameters
# ----------------------------------------------------------------
ADABOOST_CONFIG = {
    "learning_rate": 0.01,
    "max_depth": 2,
    "n_estimators_grid": [600, 700, 800, 900, 1000, 1200, 1400, 1500],
    "cv_folds": 10,
    "random_state": 1234,
}

# ----------------------------------------------------------------
# XGBoost Hyperparameters
# ----------------------------------------------------------------
XGB_CONFIG = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1,  # Will be computed based on class imbalance
    "early_stopping_rounds": 50,
    "eval_metric": "auc",
    "random_state": 1234,
    "n_jobs": -1,
}

# ----------------------------------------------------------------
# Global Random Seed
# ----------------------------------------------------------------
RANDOM_SEED = 1234
