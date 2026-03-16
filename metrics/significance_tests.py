# ================================================================
# Significance tests for comparing two classifiers on the same labels
# DeLong (AUC) and paired bootstrap (F1)
# ================================================================

import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score, f1_score


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks for DeLong structural components."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """
    DeLong covariance for unadjusted AUC (Sun & Xu 2014).
    predictions_sorted_transposed: shape (n_classifiers, n_examples), label 1 first.
    Returns (aucs 1D, delong_cov 2D).
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    if m == 0 or n == 0:
        raise ValueError("Need both positive and negative samples")
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = (tz[:, :m].sum(axis=1) / (m * n)) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(
    y_true: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    alpha: float = 0.05,
):
    """
    DeLong test for comparing two correlated AUCs on the same labels.

    Args:
        y_true: Binary labels (0 and 1).
        p1: Predicted probabilities from model 1 (same length as y_true).
        p2: Predicted probabilities from model 2.
        alpha: Significance level for the 'significant' flag.

    Returns:
        dict with: auc_1, auc_2, diff (auc_1 - auc_2), z, p_value (two-sided),
        significant (bool).
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    p1 = np.asarray(p1).ravel().astype(float)
    p2 = np.asarray(p2).ravel().astype(float)
    n = len(y_true)
    if not (len(p1) == n and len(p2) == n):
        raise ValueError("y_true, p1, p2 must have the same length")
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        return {
            "auc_1": np.nan,
            "auc_2": np.nan,
            "diff": np.nan,
            "z": np.nan,
            "p_value": np.nan,
            "significant": False,
        }
    order = (-y_true).argsort()
    label_1_count = int(np.sum(y_true == 1))
    preds = np.vstack((p1, p2))[:, order]
    try:
        aucs, delongcov = _fast_delong(preds, label_1_count)
    except ValueError:
        return {
            "auc_1": np.nan,
            "auc_2": np.nan,
            "diff": np.nan,
            "z": np.nan,
            "p_value": np.nan,
            "significant": False,
        }
    auc_1, auc_2 = float(aucs[0]), float(aucs[1])
    diff = auc_1 - auc_2
    # Variance of (AUC1 - AUC2) = var1 + var2 - 2*cov12
    var_diff = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    if var_diff <= 0:
        z = 0.0
        p_value = 1.0
    else:
        z = float(diff / np.sqrt(var_diff))
        p_value = float(2 * (1 - scipy.stats.norm.cdf(np.abs(z))))
    return {
        "auc_1": auc_1,
        "auc_2": auc_2,
        "diff": diff,
        "z": z,
        "p_value": p_value,
        "significant": p_value < alpha,
    }


def _youden_threshold(y: np.ndarray, p: np.ndarray, max_candidates: int = 200) -> float:
    """
    Threshold that maximizes Youden's J = TPR - FPR.
    Matches the threshold strategy used in evaluate_pooled().
    Uses a fast vectorized implementation suitable for bootstrap.
    """
    y = np.asarray(y).ravel().astype(int)
    p = np.asarray(p).ravel()
    if len(np.unique(y)) < 2:
        return 0.5
    # Use quantile-based candidate thresholds for speed
    thresholds = np.unique(p)
    if len(thresholds) > max_candidates:
        thresholds = np.quantile(p, np.linspace(0, 1, max_candidates))
    # Vectorized Youden's J computation
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    best_t, best_j = 0.5, -np.inf
    for t in thresholds:
        pred_pos = (p >= t)
        tp = (pred_pos & (y == 1)).sum()
        fp = (pred_pos & (y == 0)).sum()
        tpr = tp / n_pos
        fpr = fp / n_neg
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = t
    return float(best_t)


def _f1_from_proba(y_true: np.ndarray, p: np.ndarray, threshold: float = None) -> float:
    """
    Compute F1 from probabilities using a given threshold.
    If threshold is None, use Youden-optimal threshold.
    """
    y_true = np.asarray(y_true).ravel()
    p = np.asarray(p).ravel()
    if threshold is None:
        threshold = _youden_threshold(y_true, p)
    yhat = (p >= threshold).astype(int)
    return float(f1_score(y_true.astype(int), yhat, zero_division=0))


def bootstrap_f1_test(
    y_true: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 1234,
    alpha: float = 0.05,
):
    """
    Paired bootstrap test for comparing F1 of two models on the same labels.
    F1 is computed using the Youden-optimal threshold for each model
    (matching the threshold strategy used in pooled evaluation).

    Each bootstrap resample re-optimizes the threshold independently,
    ensuring a fair comparison that reflects real-world usage where
    each model picks its own best operating point.

    Args:
        y_true: Binary labels.
        p1, p2: Predicted probabilities from the two models.
        n_bootstrap: Number of bootstrap resamples.
        random_state: Random seed for resampling.
        alpha: Significance level for 'significant' and CI.

    Returns:
        dict with: f1_1, f1_2, diff (f1_1 - f1_2), p_value, ci_low, ci_high, significant.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    p1 = np.asarray(p1).ravel().astype(float)
    p2 = np.asarray(p2).ravel().astype(float)
    n = len(y_true)
    if not (len(p1) == n and len(p2) == n):
        raise ValueError("y_true, p1, p2 must have the same length")
    rng = np.random.default_rng(random_state)
    # Observed F1 with Youden-optimal threshold for each model
    f1_1 = _f1_from_proba(y_true, p1)
    f1_2 = _f1_from_proba(y_true, p2)
    diff_obs = f1_1 - f1_2
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p1_b = p1[idx]
        p2_b = p2[idx]
        f1_1_b = _f1_from_proba(y_b, p1_b)
        f1_2_b = _f1_from_proba(y_b, p2_b)
        diffs.append(f1_1_b - f1_2_b)
    diffs = np.array(diffs)
    # Two-sided p-value: proportion of bootstrap diffs on "wrong" side of 0, doubled
    p_value = float(2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0)))
    if p_value > 1.0:
        p_value = 1.0
    q_lo = (alpha / 2) * 100
    q_hi = (1 - alpha / 2) * 100
    ci_low = float(np.percentile(diffs, q_lo))
    ci_high = float(np.percentile(diffs, q_hi))
    return {
        "f1_1": f1_1,
        "f1_2": f1_2,
        "diff": diff_obs,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": p_value < alpha,
    }
