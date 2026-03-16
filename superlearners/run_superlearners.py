# ================================================================
# run_superlearners.py
# ================================================================
#
# Train RF classifiers on forecast-augmented data and evaluate.
# Reads from traintest_transforms/ (produced by data_transform.py).
#
# CLI:
#   python -m superlearners.run_superlearners \
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
from sklearn.model_selection import cross_val_score

from superlearners.config import (
    VALID_DATASETS,
    VALID_HORIZONS,
    VALID_GROUPS,
    RESULTS_DIR,
    RF_CONFIG,
    SL_MODEL_NAMES,
    MODEL_FORECASTER_MAP,
    NON_FEATURE_COLS,
    TARGET_COL,
    COUNTRY_COL,
    YEAR_COL,
    RANDOM_SEED,
    N_FOLDS,
    get_forecast_feature_names,
)
from superlearners.data_loader import (
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
    from superlearners.data_loader import np as _np
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
            from sklearn.metrics import roc_auc_score
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

    # static feature cols (same as basemodels)
    static_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS
                   and "_forecast_" not in c and "_change_" not in c]

    results = []
    y_true_dict = {}
    y_pred_dict = {}

    for model_name in SL_MODEL_NAMES:
        fc_name = MODEL_FORECASTER_MAP[model_name]
        fc_cols = get_forecast_feature_names(fc_name)
        # ensure columns exist
        fc_cols = [c for c in fc_cols if c in train_df.columns]
        feature_cols = static_cols + fc_cols

        X_train, y_train, countries_train = extract_Xy(train_df, feature_cols)
        X_test, y_test, countries_test = extract_Xy(test_df, feature_cols)

        print(f"\n  Training {model_name} ({len(feature_cols)} features)...")
        rf = train_rf(X_train, y_train, countries_train, model_name)
        preds = rf.predict_proba(X_test)
        if preds.ndim == 2:
            preds = preds[:, 1] if preds.shape[1] == 2 else np.zeros(len(X_test))

        result = evaluate_predictions(y_test, preds, model_name, year, horizon, group)
        results.append(result)
        y_true_dict[model_name] = y_test
        y_pred_dict[model_name] = preds

        # save predictions
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
        print(f"    {model_name:20s} | AUC: {auc:.4f} | F1: {f1:.4f}" if not np.isnan(auc) else
              f"    {model_name:20s} | AUC: N/A")

    return results, y_true_dict, y_pred_dict


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run superlearner RF models on augmented data")
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
    print("Superlearners - run_superlearners")
    print(f"  dataset={args.dataset}  horizon={args.horizon}  group={args.group}")
    print(f"  run_id={run_id}")
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
    md = ["# Superlearner Run Summary\n"]
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"**Runtime:** {elapsed:.1f}s\n")
    md.append(f"\n## Configuration\n")
    md.append(f"- Dataset: {args.dataset}")
    md.append(f"- Horizon: h={args.horizon}")
    md.append(f"- Group: {args.group}")
    md.append(f"- Years: {years[0]}-{years[-1]} ({len(years)} years)\n")
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
