# ================================================================
# Superlearner FAVAR-Net v2.2 Configuration (Shuffle)
# ================================================================
#
# FAVAR-Net v2.2: Enhanced NN inputs (momentum + interactions) and
# stabilized OOF via shuffled inner validation with M=3 repeats.
#
# Changes vs v2.1 (superlearner_favar):
#   - THEORY_INTERACTIONS expanded from 6 -> 14 terms
#     (adds momentum, momentum interactions, second-order terms)
#   - INNER_REPEATS = 3 for averaged shuffled OOF predictions
#
# 18 engineered features for the RF classifier (unchanged):
#   - 5 momentum features (year-over-year changes)
#   - 3 peer deviation features (country vs GroupName median)
#   - 3 basic interaction features (D*r_g, ED/FXR, CA/FXR)
#   - 6 advanced features (triple interaction, group-conditional,
#     DSED/FXR, debt acceleration, CA*VIX)
#   - 1 neural_risk_score (FAVAR-Net crisis classifier, OOF)
#
# ================================================================

import os

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_BASE = os.path.join(PROJECT_ROOT, "data", "final")
TRANSFORMS_DIR = os.path.join(PROJECT_ROOT, "superlearner_favar_shuffle", "traintest_transforms")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "superlearner_favar_shuffle", "results")
RESULTS_COMPARE_DIR = os.path.join(PROJECT_ROOT, "superlearner_favar_shuffle", "results_compare")

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
GROUP_COL = "GroupName"

# ----------------------------------------------------------------
# Macro variables used by the FAVAR-Net for lags / global factors
# ----------------------------------------------------------------
FORECAST_VARS = ["D", "ED", "DSED", "PB", "r_g", "Rem", "FXR", "lnGDPpc", "CPI", "DlnREER"]

# ----------------------------------------------------------------
# Engineered Feature Definitions
# ----------------------------------------------------------------

# Momentum: year-over-year change (delta_X = X_y - X_{y-1})
MOMENTUM_VARS = ["D", "FXR", "lnGDPpc", "r_g", "CA"]

# Peer deviation: country value minus GroupName median from train
PEER_DEV_VARS = ["D", "ED", "r_g"]

# Basic interaction features
INTERACTION_FEATURES = ["D_x_r_g", "ED_div_FXR", "CA_div_FXR"]

# Advanced features (new in v2.1)
ADVANCED_FEATURES = [
    "D_x_r_g_div_FXR",   # Triple interaction: debt sustainability / reserves
    "is_LIC",             # Binary LIC indicator
    "is_LIC_x_D",        # Group-conditional debt vulnerability
    "DSED_div_FXR",       # Short-term rollover risk
    "delta_delta_D",      # Debt acceleration (2nd derivative)
    "CA_x_lnVIX",        # CA deficit x global volatility
]

# Full list of all 18 engineered feature column names
ENGINEERED_FEATURE_NAMES = (
    [f"delta_{v}" for v in MOMENTUM_VARS]           # 5 momentum
    + [f"{v}_dev" for v in PEER_DEV_VARS]            # 3 peer deviation
    + INTERACTION_FEATURES                            # 3 basic interactions
    + ADVANCED_FEATURES                               # 6 advanced
    + ["neural_risk_score"]                           # 1 neural score
)

# ----------------------------------------------------------------
# Theory-driven interactions for FAVAR-Net INPUTS (lag vector)
# These are appended to the flattened z vector for the NN
#
# v2.2 (shuffle): expanded from 6 -> 14 terms
# ----------------------------------------------------------------
THEORY_INTERACTIONS = [
    # --- Original 6 (kept) ---
    "D_x_r_g",             # debt * borrowing cost
    "D_x_GlobalPC1",       # debt * global factor
    "CA_x_FXR",            # current account * reserves
    "r_g_x_GlobalPC1",     # borrowing cost * global factor
    "ED_x_FXR",            # external debt * reserves
    "delta_D",             # debt momentum (1st derivative)
    # --- New first-order momentum (4) ---
    "delta_FXR",           # reserve changes
    "delta_ED",            # external debt momentum
    "delta_r_g",           # borrowing cost trajectory
    "delta_CA",            # current account deterioration
    # --- New momentum interactions (2) ---
    "delta_D_x_r_g",      # debt rising * borrowing cost
    "delta_D_x_delta_FXR", # debt rising while reserves falling
    # --- New second-order (1) ---
    "delta_delta_D",       # debt acceleration
    # --- New conditional (expanded dataset only) (1) ---
    "delta_M2_GDP",        # money supply momentum (skipped if absent)
]

# ----------------------------------------------------------------
# Inner-loop repeats for shuffled validation in OOF computation
# ----------------------------------------------------------------
INNER_REPEATS = 3

# ----------------------------------------------------------------
# FAVAR-Net Configuration (tuned for v2.1, unchanged)
# ----------------------------------------------------------------
FAVAR_CONFIG = {
    "task": "classify",
    "output_dim": 1,
    "p_lags": 5,
    "n_global_factors": 2,
    "n_experts": 2,
    "residual_hidden": 64,       # was 32 -- more nonlinear capacity
    "residual_scale": 0.25,      # was 0.1 -- let GRN contribute more
    "dropout": 0.15,             # was 0.3 -- model is tiny, less regularization
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 200,
    "early_stopping_patience": 15,  # was 5 -- let it converge
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "random_state": 1234,
    "oof_folds": 10,             # was 5 -- better OOF calibration
}

# ----------------------------------------------------------------
# Random Forest Configuration (identical to basemodels)
# ----------------------------------------------------------------
RF_CONFIG = {
    "n_estimators": 1000,
    "mtry_grid": [5, 8, 10, 15, 20, 25],
    "cv_folds": 10,
    "random_state": 1234,
    "n_jobs": -1,
    "class_weight": "balanced",
}

# ----------------------------------------------------------------
# SHAP Feature Selection -- multiple K values
# ----------------------------------------------------------------
SHAP_TOP_K = [5, 10]

# ----------------------------------------------------------------
# Superlearner Model Names
# RF_FAVAR       : RF on all static + 18 engineered features
# RF_SHAP_{K}    : RF on static + top-K engineered features (SHAP)
# ----------------------------------------------------------------
SL_MODEL_NAMES = ["RF_FAVAR", "RF_SHAP_5", "RF_SHAP_10"]

# ----------------------------------------------------------------
# Cross-Validation
# ----------------------------------------------------------------
N_FOLDS = 10

# ----------------------------------------------------------------
# Global Random Seed
# ----------------------------------------------------------------
RANDOM_SEED = 1234
