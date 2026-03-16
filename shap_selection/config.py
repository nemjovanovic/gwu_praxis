# ================================================================
# SHAP Selection Configuration
# ================================================================
#
# SHAP-specific settings for computing and storing Shapley values
# from tree-based baseline models on the expanded dataset.
#
# Reuses shared constants from basemodels.config.
#
# ================================================================

import os

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHAP_RESULTS_DIR = os.path.join(PROJECT_ROOT, "shap_selection", "results")

# ----------------------------------------------------------------
# SHAP Computation
# ----------------------------------------------------------------
MAX_SHAP_SAMPLES = 500  # Subsample size for TreeExplainer (matches superlearner_favar)

# ----------------------------------------------------------------
# Models to compute SHAP for (tree-based only)
# ----------------------------------------------------------------
SHAP_MODELS = ["RandomForest", "AdaBoost", "XGBoost"]
