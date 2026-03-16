# ================================================================
# SHAP Model Configuration
# ================================================================
#
# Trains Random Forest classifiers using the top-K SHAP-selected
# features from shap_selection, then compares against baseline
# and expanded basemodels.
#
# Reuses shared constants from basemodels.config.
#
# ================================================================

import os

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "shap_model", "results")
RESULTS_COMPARE_DIR = os.path.join(PROJECT_ROOT, "shap_model", "results_compare")
SHAP_SELECTION_RESULTS_DIR = os.path.join(PROJECT_ROOT, "shap_selection", "results")

# ----------------------------------------------------------------
# SHAP Feature Selection
# ----------------------------------------------------------------
SHAP_SOURCE_MODEL = "RandomForest"
SHAP_K_VALUES = [5, 10]

# ----------------------------------------------------------------
# Model Names (one RF per K value)
# ----------------------------------------------------------------
MODEL_NAMES = [f"RF_SHAP_{k}" for k in SHAP_K_VALUES]

# ----------------------------------------------------------------
# Valid parameter ranges (same as basemodels)
# ----------------------------------------------------------------
VALID_DATASETS = ["baseline", "expanded"]
VALID_HORIZONS = [2, 5, 10]
VALID_GROUPS = ["ALL", "EME", "LIC"]
