# ================================================================
# Stacked Super Learner Configuration
# ================================================================
#
# Two-layer stacked classification ensemble:
#   Level-0: Probit, Random Forest, XGBoost, GRU-RF hybrid
#   Level-1: Logistic regression meta-classifier
#
# ================================================================

import os
import sys

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SPLITS_BASE = os.path.join(PROJECT_ROOT, "data", "final")
TRANSFORMS_DIR = os.path.join(PROJECT_ROOT, "superlearner_stacked", "traintest_transforms")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "superlearner_stacked", "results")
RESULTS_COMPARE_DIR = os.path.join(PROJECT_ROOT, "superlearner_stacked", "results_compare")

# ----------------------------------------------------------------
# Dataset / Split Configuration (same as all models)
# ----------------------------------------------------------------
VALID_DATASETS = ["baseline", "expanded"]
VALID_HORIZONS = [2, 5, 10]
VALID_GROUPS = ["ALL", "EME", "LIC"]

LOOKBACK = 10   # Rolling window size (years)
T_MIN = 2000    # First test year

def get_t_max(horizon: int) -> int:
    """Last test year for a given horizon."""
    return 2021 - horizon

# ----------------------------------------------------------------
# Column Definitions (match basemodels)
# ----------------------------------------------------------------
NON_FEATURE_COLS = [
    "WEOCountryCode",
    "GroupName",
    "Year",
    "crisis",
    "c",
    "CH",
]

TARGET_COL = "crisis"
COUNTRY_COL = "WEOCountryCode"
YEAR_COL = "Year"

# ----------------------------------------------------------------
# Level-0 Model Names
# ----------------------------------------------------------------
LEVEL0_MODEL_NAMES = ["Probit", "RandomForest", "XGBoost", "GRU_RF"]

# ----------------------------------------------------------------
# Meta-Learner (logistic only -- no NNLS)
# ----------------------------------------------------------------
META_LEARNER_NAMES = ["SL_Logistic"]

# ----------------------------------------------------------------
# Out-of-Fold CV for stacking predictions
# ----------------------------------------------------------------
OOF_FOLDS = 10  # Country-based folds, matching existing CV strategy

# ----------------------------------------------------------------
# Forecast Variables (10 key macro indicators)
# ----------------------------------------------------------------
FORECAST_VARS = ["D", "ED", "FXR", "CA", "CPI", "r_g", "PB", "Rem", "lnGDPpc", "DlnREER"]

# ----------------------------------------------------------------
# GRU Forecaster Configuration
# ----------------------------------------------------------------
FORECAST_STEPS = 2
SEQUENCE_LEN = 5

GRU_CONFIG = {
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "random_state": 1234,
}

# ----------------------------------------------------------------
# Random Forest Configuration (identical to basemodels)
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
# XGBoost Configuration (identical to basemodels)
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
    "scale_pos_weight": 1,
    "early_stopping_rounds": 50,
    "eval_metric": "auc",
    "random_state": 1234,
    "n_jobs": -1,
}

# ----------------------------------------------------------------
# Probit Configuration
# ----------------------------------------------------------------
PROBIT_CONFIG = {
    "max_iter": 1000,
    "random_state": 1234,
}

# ----------------------------------------------------------------
# Cross-Validation
# ----------------------------------------------------------------
N_FOLDS = 10

# ----------------------------------------------------------------
# Global Random Seed
# ----------------------------------------------------------------
RANDOM_SEED = 1234

# ----------------------------------------------------------------
# Helper: get GRU forecast feature column names
# ----------------------------------------------------------------
def get_forecast_feature_names(forecaster_name: str = "GRU"):
    """
    Return the forecast-feature column names for a given forecaster.
    With FORECAST_STEPS=2 and 10 vars: 10 * 2 * 2 = 40 columns.
    Naming: {var}_forecast_{step}step_{forecaster_name}
            {var}_change_{step}step_{forecaster_name}
    """
    names = []
    for var in FORECAST_VARS:
        for step in range(1, FORECAST_STEPS + 1):
            names.append(f"{var}_forecast_{step}step_{forecaster_name}")
            names.append(f"{var}_change_{step}step_{forecaster_name}")
    return names
