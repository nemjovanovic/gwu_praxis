# ================================================================
# FILE: models/metrics/evaluation.py
# ------------------------------------------------
# Unified evaluation module (per-slice + pooled overall)
# Metrics: AUC, MSE (score rule), LogLik, Precision, Recall, F1
# Includes confusion-matrix counts: TP, FP, TN, FN
# ================================================================

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# ================================================================
# Probability-based metrics (Praxis-cited)
# ================================================================

def mse_score_rule(y, p):
    """
    Scoring-rule version of MSE used in your BOI code:
      E[y(1-p)^2 + (1-y)p^2]
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    return float(np.mean(y * (1 - p) ** 2 + (1 - y) * (p ** 2)))

def safe_loglik(y, p, eps=1e-5):
    """
    Mean Bernoulli log-likelihood.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

# ================================================================
# Threshold selection
# ================================================================

def youden_threshold(y, p, max_candidates=2000):
    """
    Threshold that maximizes Youden's J = TPR - FPR.
    If y has one class, returns 0.5.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    if len(np.unique(y)) < 2:
        return 0.5

    thresholds = np.unique(p)
    if len(thresholds) > max_candidates:
        thresholds = np.quantile(p, np.linspace(0, 1, max_candidates))

    best_t, best_j = 0.5, -np.inf
    for t in thresholds:
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)

    return best_t

# ================================================================
# Per-slice evaluation (e.g., per year)
# ================================================================

def evaluate_predictions(
    y_true,
    p_hat,
    model,
    year,
    horizon,
    group,
    threshold_method="youden",
):
    """
    Evaluate predictions for a single slice (e.g., one out-of-sample year).
    Returns a dict suitable for DataFrame rows.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_hat).astype(float)

    out = {
        "Year": int(year),
        "Horizon": int(horizon),
        "Group": str(group),
        "Model": str(model),
    }

    # Threshold-free metrics (skip AUC if p contains NaN, e.g. Probit numerical issues)
    if len(np.unique(y)) == 2 and np.isfinite(p).all():
        out["AUC"] = float(roc_auc_score(y, p))
    else:
        out["AUC"] = np.nan

    out["MSE"] = mse_score_rule(y, p)
    out["LogLik"] = safe_loglik(y, p)

    # Threshold-based metrics + counts
    thr = 0.5 if threshold_method == "fixed_0.5" else youden_threshold(y, p)
    yhat = (p >= thr).astype(int)

    out["Threshold"] = float(thr)
    out["Precision"] = float(precision_score(y, yhat, zero_division=0))
    out["Recall"] = float(recall_score(y, yhat, zero_division=0))
    out["F1"] = float(f1_score(y, yhat, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out["TP"] = int(tp)
    out["FP"] = int(fp)
    out["TN"] = int(tn)
    out["FN"] = int(fn)

    return out

# ================================================================
# Pooled evaluation (De Marchi-style overall AUC)
# ================================================================

def evaluate_pooled(
    y_list,
    p_list,
    model,
    horizon,
    group,
    year_label="OVERALL",
    threshold_method="youden",
):
    """
    Pooled (overall) evaluation:
      - Concatenates all y and p across slices (years)
      - Computes pooled AUC on the concatenated sample
      - Computes totals TP/FP/TN/FN and micro precision/recall/F1
      - Computes MSE and LogLik on pooled sample (probability accuracy)

    y_list and p_list: lists of 1D arrays (one array per slice/year)
    """
    y = np.concatenate([np.asarray(v).astype(int) for v in y_list]) if len(y_list) else np.array([], dtype=int)
    p = np.concatenate([np.asarray(v).astype(float) for v in p_list]) if len(p_list) else np.array([], dtype=float)

    out = {
        "Year": str(year_label),
        "Horizon": int(horizon),
        "Group": str(group),
        "Model": str(model),
    }

    if y.size == 0 or p.size == 0:
        out.update({
            "AUC": np.nan,
            "MSE": np.nan,
            "LogLik": np.nan,
            "Threshold": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "TP": 0, "FP": 0, "TN": 0, "FN": 0
        })
        return out

    # Pooled AUC (overall AUC)
    if len(np.unique(y)) == 2:
        out["AUC"] = float(roc_auc_score(y, p))
    else:
        out["AUC"] = np.nan

    # Pooled probability metrics
    out["MSE"] = mse_score_rule(y, p)
    out["LogLik"] = safe_loglik(y, p)

    # Threshold + micro counts/metrics on pooled sample
    thr = 0.5 if threshold_method == "fixed_0.5" else youden_threshold(y, p)
    yhat = (p >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out["TP"] = int(tp)
    out["FP"] = int(fp)
    out["TN"] = int(tn)
    out["FN"] = int(fn)

    out["Threshold"] = float(thr)
    out["Precision"] = float(precision_score(y, yhat, zero_division=0))
    out["Recall"] = float(recall_score(y, yhat, zero_division=0))
    out["F1"] = float(f1_score(y, yhat, zero_division=0))

    return out
