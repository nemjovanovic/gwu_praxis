# ================================================================
# Stacked Super Learner - Main Runner
# ================================================================
#
# Two-layer stacked classification ensemble:
#
#   Level-0 models (each producing an independent crisis probability):
#     1. Probit       -- Probit regression on static features
#     2. RandomForest -- RF on static features
#     3. XGBoost      -- XGBoost on static features
#     4. GRU_RF       -- RF on static + GRU forecast features
#
#   Level-1 meta-classifier:
#     - SL_Logistic   -- L2-regularized logistic regression
#
# The meta-classifier is derived from a stratified cross-validation
# process that generates out-of-sample predictions for every
# country-year in the evaluation period.
#
# z_i = a0 + a1*p_Probit + a2*p_RF + a3*p_XGB + a4*p_GRURF
# p_SL = 1 / (1 + exp(-z_i))
#
# Usage (from the project root):
#   python -m superlearner_stacked.run_stacked \
#       --dataset baseline --horizon 2 --group ALL
#
# ================================================================

import os
import sys
import argparse
import json
import time
import shutil
import functools
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Force unbuffered output
print = functools.partial(print, flush=True)

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from superlearner_stacked.config import (
    VALID_DATASETS,
    VALID_HORIZONS,
    VALID_GROUPS,
    RESULTS_DIR,
    RF_CONFIG,
    XGB_CONFIG,
    PROBIT_CONFIG,
    RANDOM_SEED,
    N_FOLDS,
    LEVEL0_MODEL_NAMES,
    META_LEARNER_NAMES,
    OOF_FOLDS,
    NON_FEATURE_COLS,
    TARGET_COL,
    COUNTRY_COL,
    get_forecast_feature_names,
)
from superlearner_stacked.data_loader import (
    load_split,
    get_available_years,
    get_static_feature_columns,
    get_feature_columns,
    extract_Xy,
)
from metrics.evaluation import evaluate_predictions, evaluate_pooled

# Suppress warnings
warnings.filterwarnings("ignore")


# ================================================================
# CLI
# ================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stacked Super Learner: level-0 models + logistic meta-classifier"
    )
    parser.add_argument(
        "--dataset", type=str, choices=VALID_DATASETS, required=True,
        help="Dataset: 'baseline' or 'expanded'",
    )
    parser.add_argument(
        "--horizon", type=int, choices=VALID_HORIZONS, required=True,
        help="Forecast horizon: 2, 5, or 10 years",
    )
    parser.add_argument(
        "--group", type=str, choices=VALID_GROUPS, required=True,
        help="Country group: 'ALL', 'EME', or 'LIC'",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Custom run ID (default: superlearner_stacked_{dataset}_h{horizon}_{group})",
    )
    return parser.parse_args()


# ================================================================
# Directory setup
# ================================================================

def create_run_directory(run_id: str) -> str:
    """Create directory structure for run outputs (fresh each time)."""
    run_dir = os.path.join(RESULTS_DIR, run_id)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "meta_weights"), exist_ok=True)
    return run_dir


# ================================================================
# Country-based fold splitting
# ================================================================

def create_country_folds(
    countries: np.ndarray, n_folds: int, random_state: int = RANDOM_SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split samples into K folds by country (no country leaks across folds)."""
    rng = np.random.RandomState(random_state)
    unique_countries = np.unique(countries)
    rng.shuffle(unique_countries)
    country_folds = np.array_split(unique_countries, n_folds)

    folds = []
    for fold_countries in country_folds:
        val_set = set(fold_countries)
        val_mask = np.array([c in val_set for c in countries])
        folds.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    return folds


# ================================================================
# Level-0 model factories (inline, for OOF loop efficiency)
# ================================================================

def _train_probit(X, y):
    """Train Probit model (statsmodels, with sklearn fallback)."""
    try:
        import statsmodels.api as sm
        X_with_const = sm.add_constant(X, has_constant='add')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Probit(y, X_with_const)
            result = model.fit(disp=False, maxiter=PROBIT_CONFIG["max_iter"])
        return {"type": "statsmodels", "result": result}
    except Exception:
        lr = LogisticRegression(
            C=1.0, solver='lbfgs', max_iter=PROBIT_CONFIG["max_iter"],
            random_state=RANDOM_SEED,
        )
        lr.fit(X, y)
        return {"type": "sklearn", "model": lr}


def _predict_probit(model_dict, X):
    """Predict probabilities from a Probit model."""
    if model_dict["type"] == "statsmodels":
        import statsmodels.api as sm
        X_with_const = sm.add_constant(X, has_constant='add')
        proba = model_dict["result"].predict(X_with_const)
        proba = np.clip(proba.astype(np.float64), 0.0001, 0.9999)
    else:
        proba = model_dict["model"].predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            proba = proba[:, 1].astype(np.float64)
        else:
            proba = proba.ravel().astype(np.float64)
    # Replace NaN with 0.5
    if np.isnan(proba).any():
        proba = np.where(np.isnan(proba), 0.5, proba)
    return proba


def _train_rf(X, y, countries):
    """Train RF with mtry selection (matches existing basemodels RF)."""
    mtry_grid = RF_CONFIG["mtry_grid"]
    valid_mtry = [m for m in mtry_grid if m <= X.shape[1]]
    if not valid_mtry:
        valid_mtry = [max(1, int(np.sqrt(X.shape[1])))]

    # Quick CV for mtry
    folds = create_country_folds(countries, N_FOLDS, RANDOM_SEED)
    best_score, best_m = -np.inf, valid_mtry[0]
    for m in valid_mtry:
        scores = []
        for tr_idx, va_idx in folds:
            if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[va_idx])) < 2:
                continue
            rf = RandomForestClassifier(
                n_estimators=min(100, RF_CONFIG["n_estimators"]),
                max_features=m,
                class_weight=RF_CONFIG["class_weight"],
                random_state=RANDOM_SEED, n_jobs=RF_CONFIG["n_jobs"],
            )
            rf.fit(X[tr_idx], y[tr_idx])
            try:
                scores.append(roc_auc_score(y[va_idx], rf.predict_proba(X[va_idx])[:, 1]))
            except ValueError:
                pass
        if scores and np.mean(scores) > best_score:
            best_score, best_m = np.mean(scores), m

    rf_final = RandomForestClassifier(
        n_estimators=RF_CONFIG["n_estimators"],
        max_features=best_m,
        class_weight=RF_CONFIG["class_weight"],
        random_state=RANDOM_SEED, n_jobs=RF_CONFIG["n_jobs"],
    )
    rf_final.fit(X, y)
    return rf_final, best_m


def _train_xgboost(X, y, X_val=None, y_val=None):
    """Train XGBoost with early stopping."""
    n_pos = max(np.sum(y == 1), 1)
    n_neg = np.sum(y == 0)
    spw = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=XGB_CONFIG["n_estimators"],
        learning_rate=XGB_CONFIG["learning_rate"],
        max_depth=XGB_CONFIG["max_depth"],
        min_child_weight=XGB_CONFIG["min_child_weight"],
        subsample=XGB_CONFIG["subsample"],
        colsample_bytree=XGB_CONFIG["colsample_bytree"],
        gamma=XGB_CONFIG["gamma"],
        reg_alpha=XGB_CONFIG["reg_alpha"],
        reg_lambda=XGB_CONFIG["reg_lambda"],
        scale_pos_weight=spw,
        random_state=RANDOM_SEED,
        n_jobs=XGB_CONFIG["n_jobs"],
        eval_metric="auc",
        use_label_encoder=False,
    )
    if X_val is not None and y_val is not None and len(np.unique(y_val)) >= 2:
        model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X, y, verbose=False)
    return model


def _predict_proba(model, X):
    """Safely extract P(crisis) from any sklearn-like model."""
    proba = model.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1].astype(np.float64)
    return proba.ravel().astype(np.float64)


# ================================================================
# OOF prediction generation
# ================================================================

def generate_oof_predictions(
    X_static: np.ndarray,
    X_all: np.ndarray,
    y: np.ndarray,
    countries: np.ndarray,
    n_folds: int,
) -> Dict[str, np.ndarray]:
    """
    Generate out-of-fold predictions for all 4 level-0 models.

    Args:
        X_static: Feature matrix for Probit/RF/XGBoost (static features only).
        X_all: Feature matrix for GRU-RF (static + GRU forecast features).
        y: Target vector.
        countries: Country codes for fold creation.
        n_folds: Number of country-based folds.

    Returns:
        Dict mapping level-0 model name -> (n_samples,) OOF predictions.
    """
    folds = create_country_folds(countries, n_folds, RANDOM_SEED)
    n = len(y)

    oof_probit = np.zeros(n, dtype=np.float64)
    oof_rf = np.zeros(n, dtype=np.float64)
    oof_xgb = np.zeros(n, dtype=np.float64)
    oof_grurf = np.zeros(n, dtype=np.float64)

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        if len(va_idx) == 0:
            continue

        X_static_tr, y_tr = X_static[tr_idx], y[tr_idx]
        X_static_va = X_static[va_idx]
        X_all_tr = X_all[tr_idx]
        X_all_va = X_all[va_idx]
        countries_tr = countries[tr_idx]

        if len(np.unique(y_tr)) < 2:
            # Fall back to base rate
            rate = y_tr.mean()
            oof_probit[va_idx] = rate
            oof_rf[va_idx] = rate
            oof_xgb[va_idx] = rate
            oof_grurf[va_idx] = rate
            continue

        # -- Probit --
        probit_model = _train_probit(X_static_tr, y_tr)
        oof_probit[va_idx] = _predict_probit(probit_model, X_static_va)

        # -- Random Forest (on static features, use middle of mtry grid for speed) --
        valid_mtry = [m for m in RF_CONFIG["mtry_grid"] if m <= X_static_tr.shape[1]]
        if not valid_mtry:
            valid_mtry = [max(1, int(np.sqrt(X_static_tr.shape[1])))]
        best_m = valid_mtry[len(valid_mtry) // 2]
        rf = RandomForestClassifier(
            n_estimators=RF_CONFIG["n_estimators"],
            max_features=best_m,
            class_weight=RF_CONFIG["class_weight"],
            random_state=RANDOM_SEED, n_jobs=RF_CONFIG["n_jobs"],
        )
        rf.fit(X_static_tr, y_tr)
        oof_rf[va_idx] = _predict_proba(rf, X_static_va)

        # -- XGBoost (on static features, use 10% of train fold as val) --
        n_val_xgb = max(1, int(0.1 * len(X_static_tr)))
        perm = np.random.RandomState(RANDOM_SEED).permutation(len(X_static_tr))
        xtr_idx, xva_idx = perm[n_val_xgb:], perm[:n_val_xgb]
        xgb_model = _train_xgboost(
            X_static_tr[xtr_idx], y_tr[xtr_idx],
            X_static_tr[xva_idx], y_tr[xva_idx],
        )
        oof_xgb[va_idx] = _predict_proba(xgb_model, X_static_va)

        # -- GRU-RF (on static + GRU forecast features, use middle mtry) --
        valid_mtry_all = [m for m in RF_CONFIG["mtry_grid"] if m <= X_all_tr.shape[1]]
        if not valid_mtry_all:
            valid_mtry_all = [max(1, int(np.sqrt(X_all_tr.shape[1])))]
        best_m_all = valid_mtry_all[len(valid_mtry_all) // 2]
        rf_gru = RandomForestClassifier(
            n_estimators=RF_CONFIG["n_estimators"],
            max_features=best_m_all,
            class_weight=RF_CONFIG["class_weight"],
            random_state=RANDOM_SEED, n_jobs=RF_CONFIG["n_jobs"],
        )
        rf_gru.fit(X_all_tr, y_tr)
        oof_grurf[va_idx] = _predict_proba(rf_gru, X_all_va)

        print(f"      OOF fold {fold_i+1}/{n_folds}: "
              f"Probit={oof_probit[va_idx].mean():.3f}  "
              f"RF={oof_rf[va_idx].mean():.3f}  "
              f"XGB={oof_xgb[va_idx].mean():.3f}  "
              f"GRU_RF={oof_grurf[va_idx].mean():.3f}  "
              f"({len(va_idx)} held-out samples)")

    return {
        "Probit": oof_probit,
        "RandomForest": oof_rf,
        "XGBoost": oof_xgb,
        "GRU_RF": oof_grurf,
    }


# ================================================================
# Meta-learner training
# ================================================================

def train_meta_logistic(
    OOF_matrix: np.ndarray, y_train: np.ndarray,
) -> LogisticRegression:
    """
    Train L2-regularized logistic meta-learner.

    z_i = a0 + a1*p_Probit + a2*p_RF + a3*p_XGB + a4*p_GRURF
    p_SL = 1 / (1 + exp(-z_i))
    """
    lr = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
        random_state=RANDOM_SEED,
    )
    lr.fit(OOF_matrix, y_train)
    return lr


# ================================================================
# Per-year training loop
# ================================================================

def train_year(
    year: int,
    dataset: str,
    horizon: int,
    group: str,
    run_dir: str,
) -> Tuple[List[Dict], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    For a single year: generate OOF preds, train meta-learner, predict
    on test, evaluate all 5 models (4 level-0 + 1 meta).

    Returns (results_list, y_true_dict, y_pred_dict).
    """
    print(f"\n{'='*60}")
    print(f"Year {year}")
    print(f"{'='*60}")

    train_df, test_df = load_split(dataset, horizon, group, year)

    # Static feature columns (Probit, RF, XGBoost use these)
    static_cols = get_static_feature_columns(train_df)

    # All feature columns (GRU-RF uses these: static + GRU forecast features)
    all_feature_cols = get_feature_columns(train_df)

    # Extract features
    X_static_train, y_train, countries_train = extract_Xy(train_df, static_cols)
    X_all_train, _, _ = extract_Xy(train_df, all_feature_cols)
    X_static_test, y_test, countries_test = extract_Xy(test_df, static_cols)
    X_all_test, _, _ = extract_Xy(test_df, all_feature_cols)

    print(f"  Train: {len(y_train)} samples, {np.mean(y_train)*100:.1f}% crisis")
    print(f"  Test:  {len(y_test)} samples, {np.mean(y_test)*100:.1f}% crisis")
    print(f"  Static features: {X_static_train.shape[1]} cols")
    print(f"  All features (incl. GRU): {X_all_train.shape[1]} cols")

    # ------------------------------------------------------------------
    # Step A: Generate OOF predictions for all 4 level-0 models
    # ------------------------------------------------------------------
    print(f"\n  Generating OOF predictions ({OOF_FOLDS}-fold country CV)...")
    oof_preds = generate_oof_predictions(
        X_static_train, X_all_train, y_train, countries_train, OOF_FOLDS,
    )

    # Stack OOF matrix: (n_train, 4) = [Probit, RF, XGB, GRU_RF]
    OOF_matrix = np.column_stack([
        oof_preds["Probit"],
        oof_preds["RandomForest"],
        oof_preds["XGBoost"],
        oof_preds["GRU_RF"],
    ])

    print(f"\n  OOF matrix shape: {OOF_matrix.shape}")
    for i, name in enumerate(LEVEL0_MODEL_NAMES):
        col = OOF_matrix[:, i]
        try:
            auc = roc_auc_score(y_train, col)
        except ValueError:
            auc = np.nan
        print(f"    {name:20s} OOF AUC: {auc:.4f}")

    # ------------------------------------------------------------------
    # Step B: Train logistic meta-learner on OOF matrix
    # ------------------------------------------------------------------
    print(f"\n  Training logistic meta-learner...")
    logistic_meta = train_meta_logistic(OOF_matrix, y_train)

    logistic_coefs = logistic_meta.coef_.ravel()
    logistic_intercept = float(logistic_meta.intercept_[0])
    print(f"    Intercept (a0):  {logistic_intercept:.4f}")
    for name, coef in zip(LEVEL0_MODEL_NAMES, logistic_coefs):
        print(f"    {name:20s} coef: {coef:.4f}")

    # Save meta-learner weights
    weights_dir = os.path.join(run_dir, "meta_weights")
    with open(os.path.join(weights_dir, f"logistic_weights_t{year}.json"), "w") as f:
        json.dump({
            "base_learners": LEVEL0_MODEL_NAMES,
            "coefs": logistic_coefs.tolist(),
            "intercept": logistic_intercept,
        }, f, indent=2)

    # ------------------------------------------------------------------
    # Step C: Train level-0 models on FULL training set & predict test
    # ------------------------------------------------------------------
    print(f"\n  Training full level-0 models for test prediction...")

    # Probit
    probit_full = _train_probit(X_static_train, y_train)
    print(f"    Probit: trained ({probit_full['type']})")

    # Random Forest (with mtry selection)
    rf_full, rf_mtry = _train_rf(X_static_train, y_train, countries_train)
    print(f"    RandomForest: best_mtry={rf_mtry}")

    # XGBoost (with early stopping using 20% holdout)
    val_size = int(0.2 * len(y_train))
    perm = np.random.RandomState(RANDOM_SEED).permutation(len(y_train))
    xgb_full = _train_xgboost(
        X_static_train[perm[val_size:]], y_train[perm[val_size:]],
        X_static_train[perm[:val_size]], y_train[perm[:val_size]],
    )
    print(f"    XGBoost: trained")

    # GRU-RF (RF on static + GRU forecast features, with mtry selection)
    grurf_full, grurf_mtry = _train_rf(X_all_train, y_train, countries_train)
    print(f"    GRU_RF: best_mtry={grurf_mtry}")

    # Test predictions from each level-0 model
    test_probit = _predict_probit(probit_full, X_static_test)
    test_rf = _predict_proba(rf_full, X_static_test)
    test_xgb = _predict_proba(xgb_full, X_static_test)
    test_grurf = _predict_proba(grurf_full, X_all_test)

    # Stack test matrix: (n_test, 4)
    test_matrix = np.column_stack([test_probit, test_rf, test_xgb, test_grurf])

    # ------------------------------------------------------------------
    # Step D: Meta-learner test predictions
    # ------------------------------------------------------------------
    test_sl_logistic = logistic_meta.predict_proba(test_matrix)[:, 1]

    # ------------------------------------------------------------------
    # Step E: Evaluate all 5 models (4 level-0 + 1 meta)
    # ------------------------------------------------------------------
    all_test_preds = {
        "Probit": test_probit,
        "RandomForest": test_rf,
        "XGBoost": test_xgb,
        "GRU_RF": test_grurf,
        "SL_Logistic": test_sl_logistic,
    }

    results = []
    y_true_dict = {}
    y_pred_dict = {}

    for model_name, preds in all_test_preds.items():
        result = evaluate_predictions(y_test, preds, model_name, year, horizon, group)
        results.append(result)
        y_true_dict[model_name] = y_test
        y_pred_dict[model_name] = preds

        # Save predictions
        pred_dir = os.path.join(run_dir, "predictions", model_name)
        os.makedirs(pred_dir, exist_ok=True)
        pred_df = pd.DataFrame({
            "Year": np.full(len(y_test), year, dtype=np.int64),
            "WEOCountryCode": countries_test,
            "y_true": y_test,
            "y_pred": preds,
        })
        pred_df.to_csv(os.path.join(pred_dir, f"preds_t{year}.csv"), index=False)

        auc = result.get("AUC", np.nan)
        f1 = result.get("F1", np.nan)
        auc_s = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        f1_s = f"{f1:.4f}" if not np.isnan(f1) else "N/A"
        print(f"    {model_name:20s} | AUC: {auc_s} | F1: {f1_s}")

    return results, y_true_dict, y_pred_dict


# ================================================================
# Pooled evaluation
# ================================================================

def compute_pooled_results(
    all_y_true: Dict[str, List[np.ndarray]],
    all_y_pred: Dict[str, List[np.ndarray]],
    horizon: int,
    group: str,
) -> List[Dict]:
    """Compute pooled (overall) evaluation metrics for all models."""
    results = []
    for model_name in all_y_true:
        result = evaluate_pooled(
            all_y_true[model_name], all_y_pred[model_name],
            model_name, horizon, group,
        )
        results.append(result)
    return results


# ================================================================
# Summary markdown
# ================================================================

def generate_summary_markdown(
    run_dir: str,
    config: Dict,
    results_by_year: pd.DataFrame,
    results_overall: pd.DataFrame,
    runtime_seconds: float,
) -> str:
    """Generate markdown summary with results and meta-learner weights."""
    md = []

    md.append("# Stacked Super Learner Run Summary\n")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"**Runtime:** {runtime_seconds:.1f} seconds\n")

    # Configuration
    md.append("\n## Configuration\n")
    md.append(f"- **Dataset:** {config['dataset']}")
    md.append(f"- **Horizon:** h={config['horizon']} years")
    md.append(f"- **Group:** {config['group']}")
    md.append(f"- **Years:** {config['years'][0]} - {config['years'][-1]} ({len(config['years'])} years)")
    md.append(f"- **OOF Folds:** {OOF_FOLDS} (country-based)")
    md.append(f"- **Level-0 Models:** {', '.join(LEVEL0_MODEL_NAMES)}")
    md.append(f"- **Meta-Learner:** {', '.join(META_LEARNER_NAMES)}")
    md.append(f"- **Random Seed:** {RANDOM_SEED}\n")

    # Overall Results
    md.append("\n## Overall Results (Pooled)\n")
    md.append("| Model | AUC | MSE | LogLik | Precision | Recall | F1 |")
    md.append("|-------|-----|-----|--------|-----------|--------|-----|")

    # Sort: meta-learner first, then level-0 models
    model_order = META_LEARNER_NAMES + LEVEL0_MODEL_NAMES
    results_overall = results_overall.copy()
    results_overall["_sort"] = results_overall["Model"].apply(
        lambda x: model_order.index(x) if x in model_order else 99
    )
    results_sorted = results_overall.sort_values("_sort").drop("_sort", axis=1)

    for _, row in results_sorted.iterrows():
        vals = []
        for col in ["AUC", "MSE", "LogLik", "Precision", "Recall", "F1"]:
            v = row.get(col, np.nan)
            vals.append(f"{v:.4f}" if not pd.isna(v) else "N/A")
        md.append(f"| {row['Model']} | {' | '.join(vals)} |")
    md.append("")

    # Meta-learner weights (averaged across years)
    md.append("\n## Meta-Learner Weights (Averaged Across Years)\n")

    weights_dir = os.path.join(run_dir, "meta_weights")
    pooled_log_path = os.path.join(weights_dir, "logistic_weights_pooled.json")

    if os.path.isfile(pooled_log_path):
        with open(pooled_log_path) as f:
            log_data = json.load(f)
        md.append("### Logistic Meta-Classifier\n")
        md.append("z_i = a0 + a1\\*p_Probit + a2\\*p_RF + a3\\*p_XGB + a4\\*p_GRURF\n")
        md.append("| Parameter | Value |")
        md.append("|-----------|-------|")
        md.append(f"| Intercept (a0) | {log_data['intercept']:.4f} |")
        for name, c in zip(log_data["base_learners"], log_data["coefs"]):
            md.append(f"| {name} | {c:.4f} |")
        md.append("")

    # Per-Year AUC
    md.append("\n## Per-Year AUC\n")
    auc_pivot = results_by_year.pivot(index="Year", columns="Model", values="AUC")
    desired = [m for m in model_order if m in auc_pivot.columns]
    auc_pivot = auc_pivot[desired]
    cols = auc_pivot.columns.tolist()
    md.append("| Year | " + " | ".join(cols) + " |")
    md.append("|------|" + "|".join(["------"] * len(cols)) + "|")
    for yr, row in auc_pivot.iterrows():
        vals = [f"{v:.4f}" if not pd.isna(v) else "N/A" for v in row.values]
        md.append(f"| {yr} | " + " | ".join(vals) + " |")
    md.append("")

    # Output files
    md.append("\n## Output Files\n")
    md.append("- Results by year: `results_by_year.csv`")
    md.append("- Overall results: `results_overall.csv`")
    md.append("- Predictions: `predictions/{Model}/`")
    md.append("- Meta-learner weights: `meta_weights/`")
    md.append("")

    return "\n".join(md)


# ================================================================
# Main
# ================================================================

def main():
    """Main entry point."""
    args = parse_args()

    run_id = args.run_id or f"superlearner_stacked_{args.dataset}_h{args.horizon}_{args.group}"

    print("=" * 60)
    print("STACKED SUPER LEARNER")
    print("=" * 60)
    print(f"Dataset:        {args.dataset}")
    print(f"Horizon:        h={args.horizon}")
    print(f"Group:          {args.group}")
    print(f"Level-0 Models: {', '.join(LEVEL0_MODEL_NAMES)}")
    print(f"Meta-Learner:   {', '.join(META_LEARNER_NAMES)}")
    print(f"OOF Folds:      {OOF_FOLDS}")
    print(f"Run ID:         {run_id}")
    print("=" * 60)

    start_time = time.time()

    # Create run directory
    run_dir = create_run_directory(run_id)
    print(f"\nOutput directory: {run_dir}")

    # Discover available years from augmented data
    years = get_available_years(args.dataset, args.horizon, args.group)
    if not years:
        print("ERROR: No augmented splits found in superlearner_stacked/traintest_transforms/.")
        print("Run data_transform first:")
        print(f"  python -m superlearner_stacked.data_transform --dataset {args.dataset} "
              f"--horizon {args.horizon} --group {args.group}")
        return

    print(f"Available years: {years[0]} - {years[-1]} ({len(years)} years)")

    # Save config
    config = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "group": args.group,
        "run_id": run_id,
        "years": years,
        "oof_folds": OOF_FOLDS,
        "level0_models": LEVEL0_MODEL_NAMES,
        "meta_learners": META_LEARNER_NAMES,
        "random_seed": RANDOM_SEED,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Train and evaluate for each year
    all_results: List[Dict] = []
    all_y_true: Dict[str, List[np.ndarray]] = {}
    all_y_pred: Dict[str, List[np.ndarray]] = {}

    for year in years:
        year_results, y_true_dict, y_pred_dict = train_year(
            year, args.dataset, args.horizon, args.group, run_dir,
        )
        all_results.extend(year_results)

        for mn in y_true_dict:
            all_y_true.setdefault(mn, []).append(y_true_dict[mn])
            all_y_pred.setdefault(mn, []).append(y_pred_dict[mn])

    # Pooled prediction files
    for mn in all_y_true:
        pred_dir = os.path.join(run_dir, "predictions", mn)
        year_files = sorted(
            f for f in os.listdir(pred_dir)
            if f.startswith("preds_t") and f.endswith(".csv")
        )
        frames = [pd.read_csv(os.path.join(pred_dir, f)) for f in year_files]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                os.path.join(pred_dir, "preds_pooled.csv"), index=False
            )

    # Save per-year results
    results_by_year = pd.DataFrame(all_results)
    results_by_year.to_csv(os.path.join(run_dir, "results_by_year.csv"), index=False)

    # Pooled evaluation
    pooled_results = compute_pooled_results(
        all_y_true, all_y_pred, args.horizon, args.group,
    )
    results_overall = pd.DataFrame(pooled_results)
    results_overall.to_csv(os.path.join(run_dir, "results_overall.csv"), index=False)

    # Pooled meta-learner weights (average across years)
    weights_dir = os.path.join(run_dir, "meta_weights")

    log_coefs_all = []
    log_intercepts = []
    for year in years:
        log_path = os.path.join(weights_dir, f"logistic_weights_t{year}.json")
        if os.path.isfile(log_path):
            with open(log_path) as f:
                d = json.load(f)
                log_coefs_all.append(d["coefs"])
                log_intercepts.append(d["intercept"])

    if log_coefs_all:
        avg_coefs = np.mean(log_coefs_all, axis=0).tolist()
        avg_intercept = float(np.mean(log_intercepts))
        with open(os.path.join(weights_dir, "logistic_weights_pooled.json"), "w") as f:
            json.dump({
                "base_learners": LEVEL0_MODEL_NAMES,
                "coefs": avg_coefs,
                "intercept": avg_intercept,
            }, f, indent=2)

    # Runtime
    runtime_seconds = time.time() - start_time

    # Summary markdown
    summary_md = generate_summary_markdown(
        run_dir, config, results_by_year, results_overall, runtime_seconds,
    )
    with open(os.path.join(run_dir, "run_summary.md"), "w") as f:
        f.write(summary_md)

    # Print final summary
    print("\n" + "=" * 60)
    print("OVERALL RESULTS (Pooled)")
    print("=" * 60)
    print(f"{'Model':20s} | {'AUC':8s} | {'MSE':8s} | {'F1':8s}")
    print("-" * 55)

    model_order = META_LEARNER_NAMES + LEVEL0_MODEL_NAMES
    results_overall = results_overall.copy()
    results_overall["_sort"] = results_overall["Model"].apply(
        lambda x: model_order.index(x) if x in model_order else 99
    )
    results_overall = results_overall.sort_values("_sort").drop("_sort", axis=1)

    for _, row in results_overall.iterrows():
        auc = f"{row['AUC']:.4f}" if not pd.isna(row['AUC']) else "N/A"
        mse = f"{row['MSE']:.4f}" if not pd.isna(row['MSE']) else "N/A"
        f1 = f"{row['F1']:.4f}" if not pd.isna(row['F1']) else "N/A"
        print(f"{row['Model']:20s} | {auc:8s} | {mse:8s} | {f1:8s}")

    # Print meta-learner weights
    if log_coefs_all:
        print(f"\nLogistic meta-classifier (pooled):")
        print(f"  Intercept (a0): {avg_intercept:.4f}")
        for name, c in zip(LEVEL0_MODEL_NAMES, avg_coefs):
            print(f"  {name:20s}: {c:.4f}")

    print("\n" + "=" * 60)
    print(f"Run completed in {runtime_seconds:.1f} seconds")
    print(f"Results saved to: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
