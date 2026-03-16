# ================================================================
# Base Models Package
# ================================================================
#
# Models:
#   - Rule of Thumb (ROT): Historical crisis rate baseline
#   - Probit: GLM with probit link
#   - Random Forest: With mtry tuning (K=10 CV)
#   - AdaBoost: GBM with adaboost distribution
#   - XGBoost: Gradient boosting model
#
# Usage:
#   python -m basemodels.run_basemodels --dataset baseline --horizon 2 --group ALL
#
# ================================================================

from .models import (
    RuleOfThumbModel,
    ProbitModel,
    RandomForestModel,
    AdaBoostModel,
    XGBoostModel,
)

__all__ = [
    "RuleOfThumbModel",
    "ProbitModel",
    "RandomForestModel",
    "AdaBoostModel",
    "XGBoostModel",
]
