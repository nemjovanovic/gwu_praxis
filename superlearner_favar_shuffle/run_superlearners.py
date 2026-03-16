# ================================================================
# run_superlearners.py (Superlearner FAVAR-Net v2.2 - Shuffle)
# ================================================================
#
# Train RF classifiers on FAVAR-Net v2.2 feature-augmented data, with
# SHAP-based feature selection at multiple K values.
#
# Features: 18 engineered columns (5 momentum + 3 peer deviation +
# 3 basic interactions + 6 advanced + 1 neural_risk_score)
# on top of static features.
#
# Models:
#   RF_FAVAR   - RF on all static + 18 engineered features
#   RF_SHAP_5  - RF on static + top-5 engineered features (SHAP)
#   RF_SHAP_10 - RF on static + top-10 engineered features (SHAP)
#
# CLI:
#   python -m superlearner_favar_shuffle.run_superlearners \
#       --dataset baseline --horizon 2 --group ALL
# ================================================================

import os
import sys
import argparse
import json
import time
import functools
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import shap

from superlearner_favar_shuffle.config import (
    VALID_DATASETS,
    VALID_HORIZONS,
    VALID_GROUPS,
    RESULTS_DIR,
    RF_CONFIG,
    SL_MODEL_NAMES,
    SHAP_TOP_K,
    NON_FEATURE_COLS,
    TARGET_COL,
    COUNTRY_COL,
    YEAR_COL,
    RANDOM_SEED,
    N_FOLDS,
    ENGINEERED_FEATURE_NAMES,
)
from superlearner_favar_shuffle.data_loader import (
    load_split,
    get_available_years,
    get_feature_columns,
    extract_Xy,
)
from metrics.evaluation import evaluate_predictions, evaluate_pooled

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------
# RF with mtry selection (same as basemodels RandomForest)
# ----------------------------------------------------------------

def _select_mtry(X, y, countries, mtry_grid, n_folds, random_state, n_jobs):
    """Country-based CV to pick best max_features."""
    unique_countries = np.unique(countries)
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_countries)
    folds = np.array_split(unique_countries, n_folds)

    fold_indices = []
    for f in folds:
        val_set = set(f)
        val_mask = np.array([c in val_set for c in countries])
        fold_indices.append((np.where(~val_mask)[0], np.where(val_mask)[0]))

    valid_mtry = [m for m in mtry_grid if m <= X.shape[1]]
    if not valid_mtry:
        valid_mtry = [max(1, int(np.sqrt(X.shape[1])))]

    best_score, best_m = -np.inf, valid_mtry[0]
    for m in valid_mtry:
        scores = []
        for train_idx, val_idx in fold_indices:
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
                continue
            rf = RandomForestClassifier(
                n_estimators=min(100, RF_CONFIG["n_estimators"]),
                max_features=m,
                class_weight=RF_CONFIG["class_weight"],
                random_state=random_state,
                n_jobs=n_jobs,
            )
            rf.fit(X[train_idx], y[train_idx])
            try:
                scores.append(roc_auc_score(y[val_idx], rf.predict_proba(X[val_idx])[:, 1]))
            except ValueError:
                pass
        if scores and np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_m = m
    return best_m


def train_rf(X_train, y_train, countries_train, model_name):
    """Train an RF classifier with mtry selection (same as basemodels)."""
    best_mtry = _select_mtry(
        X_train, y_train, countries_train,
        RF_CONFIG["mtry_grid"], N_FOLDS, RF_CONFIG["random_state"], RF_CONFIG["n_jobs"],
    )
    rf = RandomForestClassifier(
        n_estimators=RF_CONFIG["n_estimators"],
        max_features=best_mtry,
        class_weight=RF_CONFIG["class_weight"],
        random_state=RF_CONFIG["random_state"],
        n_jobs=RF_CONFIG["n_jobs"],
    )
    rf.fit(X_train, y_train)
    print(f"    {model_name}: best_mtry={best_mtry}")
    return rf


# ----------------------------------------------------------------
# SHAP computation and persistence
# ----------------------------------------------------------------

def compute_shap_values(rf_model, X_train, feature_cols):
    """Compute SHAP values for a trained RF on (a subsample of) X_train."""
    max_shap_samples = 500
    if X_train.shape[0] > max_shap_samples:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(X_train.shape[0], max_shap_samples, replace=False)
        X_shap = X_train[idx]
    else:
        X_shap = X_train

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        sv_signed = shap_values[1]
    elif shap_values.ndim == 3:
        sv_signed = shap_values[:, :, 1]
    else:
        sv_signed = shap_values

    sv_signed = np.asarray(sv_signed, dtype=np.float32)
    mean_abs_shap = np.abs(sv_signed).mean(axis=0)

    return sv_signed, X_shap, mean_abs_shap


def save_shap_artifacts(sv_signed, X_shap, mean_abs_shap, feature_cols, save_dir, year):
    """Persist SHAP artifacts to disk for later plotting."""
    os.makedirs(save_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(save_dir, f"shap_values_t{year}.npz"),
        shap_values=sv_signed.astype(np.float32),
        X_shap=X_shap.astype(np.float32),
        feature_names=np.array(feature_cols, dtype=object),
    )

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df.to_csv(
        os.path.join(save_dir, f"shap_importance_t{year}.csv"), index=False,
    )

    print(f"    SHAP artifacts saved to {save_dir}")


def select_top_k_from_shap(mean_abs_shap, feature_cols, engineered_cols, k):
    """
    Given pre-computed mean |SHAP| values, select top-K *engineered* features.
    """
    eng_set = set(engineered_cols)
    eng_importance = []
    for i, col in enumerate(feature_cols):
        if col in eng_set:
            eng_importance.append((col, mean_abs_shap[i]))

    eng_importance.sort(key=lambda x: x[1], reverse=True)
    k_actual = min(k, len(eng_importance))
    selected = [name for name, _ in eng_importance[:k_actual]]

    print(f"    SHAP top-{k_actual} engineered features selected")
    for name, imp in eng_importance[:k_actual]:
        print(f"      {name:30s} | mean|SHAP|={imp:.6f}")

    return selected


# ----------------------------------------------------------------
# per-year loop
# ----------------------------------------------------------------

def train_year(
    year: int, dataset: str, horizon: int, group: str, run_dir: str,
) -> Tuple[List[Dict], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Train all SL models for one year, save predictions, return results."""
    print(f"\n{'='*60}")
    print(f"  Year {year}")
    print(f"{'='*60}")

    train_df, test_df = load_split(dataset, horizon, group, year)

    # Identify the engineered feature columns present in this data
    eng_cols_present = [c for c in ENGINEERED_FEATURE_NAMES if c in train_df.columns]

    # Static feature cols = everything except non-feature cols and engineered cols
    eng_set = set(ENGINEERED_FEATURE_NAMES)
    static_cols = [c for c in train_df.columns
                   if c not in NON_FEATURE_COLS and c not in eng_set]

    results = []
    y_true_dict = {}
    y_pred_dict = {}

    # ----- 1) Train RF_FAVAR on ALL static + engineered features -----
    all_feature_cols = static_cols + eng_cols_present
    X_train_all, y_train, countries_train = extract_Xy(train_df, all_feature_cols)
    X_test_all, y_test, countries_test = extract_Xy(test_df, all_feature_cols)

    print(f"\n  Training RF_FAVAR ({len(all_feature_cols)} features: "
          f"{len(static_cols)} static + {len(eng_cols_present)} engineered)...")
    rf_favar = train_rf(X_train_all, y_train, countries_train, "RF_FAVAR")
    preds_favar = rf_favar.predict_proba(X_test_all)
    if preds_favar.ndim == 2:
        preds_favar = preds_favar[:, 1] if preds_favar.shape[1] == 2 else np.zeros(len(X_test_all))

    result_favar = evaluate_predictions(y_test, preds_favar, "RF_FAVAR", year, horizon, group)
    results.append(result_favar)
    y_true_dict["RF_FAVAR"] = y_test
    y_pred_dict["RF_FAVAR"] = preds_favar

    # save RF_FAVAR predictions
    pred_dir = os.path.join(run_dir, "predictions", "RF_FAVAR")
    os.makedirs(pred_dir, exist_ok=True)
    pred_df = pd.DataFrame({
        "Year": np.full(len(y_test), year, dtype=np.int64),
        "WEOCountryCode": countries_test,
        "y_true": y_test,
        "y_pred": preds_favar,
    })
    pred_df.to_csv(os.path.join(pred_dir, f"preds_t{year}.csv"), index=False)

    auc = result_favar.get("AUC", np.nan)
    f1 = result_favar.get("F1", np.nan)
    print(f"    {'RF_FAVAR':20s} | AUC: {auc:.4f} | F1: {f1:.4f}" if not np.isnan(auc) else
          f"    {'RF_FAVAR':20s} | AUC: N/A")

    # ----- 2) Compute SHAP values once (from RF_FAVAR) and save -----
    print(f"\n  Computing SHAP values from RF_FAVAR...")
    sv_signed, X_shap, mean_abs_shap = compute_shap_values(
        rf_favar, X_train_all, all_feature_cols,
    )

    shap_dir = os.path.join(run_dir, "shap")
    save_shap_artifacts(sv_signed, X_shap, mean_abs_shap, all_feature_cols, shap_dir, year)

    # ----- 3) SHAP-based feature selection for each K -----
    for k in SHAP_TOP_K:
        model_name = f"RF_SHAP_{k}"
        print(f"\n  Training {model_name} (static + top-{k} engineered features)...")

        selected_eng = select_top_k_from_shap(
            mean_abs_shap, all_feature_cols, eng_cols_present, k,
        )
        shap_feature_cols = static_cols + selected_eng

        X_train_shap, _, _ = extract_Xy(train_df, shap_feature_cols)
        X_test_shap, _, _ = extract_Xy(test_df, shap_feature_cols)

        rf_shap = train_rf(X_train_shap, y_train, countries_train, model_name)
        preds_shap = rf_shap.predict_proba(X_test_shap)
        if preds_shap.ndim == 2:
            preds_shap = preds_shap[:, 1] if preds_shap.shape[1] == 2 else np.zeros(len(X_test_shap))

        result_shap = evaluate_predictions(y_test, preds_shap, model_name, year, horizon, group)
        results.append(result_shap)
        y_true_dict[model_name] = y_test
        y_pred_dict[model_name] = preds_shap

        # save predictions
        pred_dir_shap = os.path.join(run_dir, "predictions", model_name)
        os.makedirs(pred_dir_shap, exist_ok=True)
        pred_df_shap = pd.DataFrame({
            "Year": np.full(len(y_test), year, dtype=np.int64),
            "WEOCountryCode": countries_test,
            "y_true": y_test,
            "y_pred": preds_shap,
        })
        pred_df_shap.to_csv(os.path.join(pred_dir_shap, f"preds_t{year}.csv"), index=False)

        # save which features were selected
        sel_path = os.path.join(pred_dir_shap, f"selected_features_t{year}.json")
        with open(sel_path, "w") as f:
            json.dump({"k": k, "selected_engineered_features": selected_eng}, f, indent=2)

        auc_s = result_shap.get("AUC", np.nan)
        f1_s = result_shap.get("F1", np.nan)
        print(f"    {model_name:20s} | AUC: {auc_s:.4f} | F1: {f1_s:.4f}" if not np.isnan(auc_s) else
              f"    {model_name:20s} | AUC: N/A")

    return results, y_true_dict, y_pred_dict


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run superlearner FAVAR-Net v2.2 (shuffle) RF models with SHAP feature selection")
    p.add_argument("--dataset", type=str, choices=VALID_DATASETS, required=True)
    p.add_argument("--horizon", type=int, choices=VALID_HORIZONS, required=True)
    p.add_argument("--group", type=str, choices=VALID_GROUPS, required=True)
    p.add_argument("--run-id", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    start = time.time()

    run_id = args.run_id or f"superlearner_{args.dataset}_h{args.horizon}_{args.group}"
    run_dir = os.path.join(RESULTS_DIR, run_id)
    # Delete existing results for this combination so they are always fresh
    if os.path.exists(run_dir):
        import shutil
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)

    print("=" * 60)
    print("Superlearner FAVAR-Net v2.2 (shuffle) - run_superlearners")
    print(f"  dataset={args.dataset}  horizon={args.horizon}  group={args.group}")
    print(f"  run_id={run_id}")
    print(f"  models: {SL_MODEL_NAMES}")
    print(f"  SHAP_TOP_K: {SHAP_TOP_K}")
    print(f"  Engineered features: {ENGINEERED_FEATURE_NAMES}")
    print("=" * 60)

    years = get_available_years(args.dataset, args.horizon, args.group)
    if not years:
        print("ERROR: No augmented splits found. Run data_transform first.")
        return
    print(f"Available years: {years}")

    # save config
    config = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "group": args.group,
        "years": years,
        "models": SL_MODEL_NAMES,
        "shap_top_k": SHAP_TOP_K,
        "engineered_features": ENGINEERED_FEATURE_NAMES,
        "rf_config": RF_CONFIG,
        "random_seed": RANDOM_SEED,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    all_results = []
    all_y_true: Dict[str, List[np.ndarray]] = {}
    all_y_pred: Dict[str, List[np.ndarray]] = {}

    for year in years:
        yr, yt, yp = train_year(year, args.dataset, args.horizon, args.group, run_dir)
        all_results.extend(yr)
        for mn in yt:
            all_y_true.setdefault(mn, []).append(yt[mn])
            all_y_pred.setdefault(mn, []).append(yp[mn])

    # pooled results
    print("\n" + "=" * 60)
    print("Pooled Results")
    print("=" * 60)
    pooled_results = []
    for mn in all_y_true:
        r = evaluate_pooled(all_y_true[mn], all_y_pred[mn], mn, args.horizon, args.group)
        pooled_results.append(r)
        auc = r.get("AUC", np.nan)
        f1 = r.get("F1", np.nan)
        print(f"  {mn:20s} | AUC: {auc:.4f} | F1: {f1:.4f}")

    # build preds_pooled per model
    for mn in all_y_true:
        pred_dir = os.path.join(run_dir, "predictions", mn)
        year_files = sorted(
            [f for f in os.listdir(pred_dir) if f.startswith("preds_t") and f.endswith(".csv")]
        )
        frames = [pd.read_csv(os.path.join(pred_dir, f)) for f in year_files]
        if frames:
            pooled_df = pd.concat(frames, ignore_index=True)
            pooled_df.to_csv(os.path.join(pred_dir, "preds_pooled.csv"), index=False)

    # save results CSVs
    pd.DataFrame(all_results).to_csv(os.path.join(run_dir, "results_by_year.csv"), index=False)
    pd.DataFrame(pooled_results).to_csv(os.path.join(run_dir, "results_overall.csv"), index=False)

    elapsed = time.time() - start

    # summary markdown
    md = ["# Superlearner FAVAR-Net v2.2 (Shuffle) Run Summary\n"]
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"**Runtime:** {elapsed:.1f}s\n")
    md.append(f"\n## Configuration\n")
    md.append(f"- Dataset: {args.dataset}")
    md.append(f"- Horizon: h={args.horizon}")
    md.append(f"- Group: {args.group}")
    md.append(f"- Years: {years[0]}-{years[-1]} ({len(years)} years)")
    md.append(f"- SHAP Top-K: {SHAP_TOP_K}")
    md.append(f"- Engineered features ({len(ENGINEERED_FEATURE_NAMES)}): {ENGINEERED_FEATURE_NAMES}\n")
    md.append(f"\n## Pooled Results\n")
    md.append("| Model | AUC | F1 | Precision | Recall |")
    md.append("|-------|-----|----|-----------| -------|")
    for r in sorted(pooled_results, key=lambda x: x.get("AUC", 0), reverse=True):
        md.append(
            f"| {r['Model']} | {r.get('AUC', np.nan):.4f} | {r.get('F1', np.nan):.4f} "
            f"| {r.get('Precision', np.nan):.4f} | {r.get('Recall', np.nan):.4f} |"
        )
    md.append("")
    with open(os.path.join(run_dir, "run_summary.md"), "w") as f:
        f.write("\n".join(md))

    print(f"\nResults saved to: {run_dir}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
