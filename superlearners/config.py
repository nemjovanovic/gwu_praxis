# ================================================================
# Superlearners Configuration
# ================================================================
#
# Forecast-augmented RF: LSTM / PatchTST forecasters -> features -> RF
# ================================================================

import os

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_BASE = os.path.join(PROJECT_ROOT, "data", "final")
TRANSFORMS_DIR = os.path.join(PROJECT_ROOT, "superlearners", "traintest_transforms")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "superlearners", "results")
RESULTS_COMPARE_DIR = os.path.join(PROJECT_ROOT, "superlearners", "results_compare")

# ----------------------------------------------------------------
# Dataset / Split Configuration (same as basemodels)
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
# Forecast Variables (10 key macro indicators)
# ----------------------------------------------------------------
FORECAST_VARS = ["D", "ED", "FXR", "CA", "CPI", "r_g", "PB", "Rem", "lnGDPpc", "DlnREER"]

# ----------------------------------------------------------------
# Sequence length for forecasters
# ----------------------------------------------------------------
SEQUENCE_LEN = 5

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
# LSTM Forecaster Configuration
# ----------------------------------------------------------------
LSTM_CONFIG = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "random_state": 1234,
}

# ----------------------------------------------------------------
# PatchTST Configuration
# ----------------------------------------------------------------
PATCHTST_CONFIG = {
    "patch_len": 2,
    "stride": 2,
    "d_model": 32,
    "n_heads": 2,
    "n_layers": 2,
    "d_ff": 64,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "random_state": 1234,
}

# ----------------------------------------------------------------
# Superlearner Model Names (RF variants with forecast features)
# ----------------------------------------------------------------
SL_MODEL_NAMES = ["RF_LSTM", "RF_PatchTST"]

# Map model name -> forecaster name used in column suffixes
MODEL_FORECASTER_MAP = {
    "RF_LSTM": "LSTM",
    "RF_PatchTST": "PatchTST",
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
# Helper: get forecast feature column names for a given forecaster
# ----------------------------------------------------------------
def get_forecast_feature_names(forecaster_name: str):
    """Return the 2*len(FORECAST_VARS) forecast-feature column names for one forecaster."""
    names = []
    for var in FORECAST_VARS:
        names.append(f"{var}_forecast_{forecaster_name}")
        names.append(f"{var}_change_{forecaster_name}")
    return names
