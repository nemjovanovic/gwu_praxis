# ================================================================
# SHAP Model - Main Runner
# ================================================================
#
# Reads SHAP feature importance rankings from shap_selection,
# selects the top-K features, trains Random Forest classifiers
# on the expanded dataset using only those features, and saves
# predictions and evaluation results.
#
# Usage (from the project root):
#   python -m shap_model.run_shap_model --horizon 2 --group ALL
#   python -m shap_model.run_shap_model --horizon 5 --group EME
#   python -m shap_model.run_shap_model --horizon 10 --group LIC
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
from typing import Dict, List, Tuple

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from basemodels.config import (
    VALID_HORIZONS,
    VALID_GROUPS,
    N_FOLDS,
    RANDOM_SEED,
    RF_CONFIG,
)
from basemodels.data_loader import DataLoader, extract_Xy
from basemodels.models import RandomForestModel
from metrics.evaluation import evaluate_predictions, evaluate_pooled

from shap_model.config import (
    RESULTS_DIR,
    SHAP_SELECTION_RESULTS_DIR,
    SHAP_SOURCE_MODEL,
    SHAP_K_VALUES,
    MODEL_NAMES,
)

warnings.filterwarnings("ignore")


# ================================================================
# CLI
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RF classifiers on SHAP-selected features"
    )
    parser.add_argument(
        "--horizon", type=int, choices=VALID_HORIZONS, required=True,
        help="Forecast horizon: 2, 5, or 10 years",
    )
    parser.add_argument(
        "--group", type=str, choices=VALID_GROUPS, required=True,
        help="Country group: ALL, EME, or LIC",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Custom run ID (default: shap_h{horizon}_{group})",
    )
    return parser.parse_args()


# ================================================================
# SHAP feature loading
# ================================================================

def load_shap_features(horizon: int, group: str) -> pd.DataFrame:
    """
    Load pooled SHAP importance rankings from shap_selection results.

    Returns:
        DataFrame with columns [feature, mean_abs_shap, rank] sorted by rank.
    """
    shap_run = f"expanded_h{horizon}_{group}"
    path = os.path.join(
        SHAP_SELECTION_RESULTS_DIR, shap_run, "shap",
        SHAP_SOURCE_MODEL, "shap_importance_pooled.csv",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"SHAP importance file not found: {path}\n"
            f"Run shap_selection first: python -m shap_selection.run_shap --horizon {horizon} --group {group}"
        )
    df = pd.read_csv(path).sort_values("rank").reset_index(drop=True)
    return df


def get_top_k_features(shap_df: pd.DataFrame, k: int) -> List[str]:
    """Return the top-k feature names from the SHAP importance ranking."""
    return shap_df.head(k)["feature"].tolist()


# ================================================================
# Directory setup
# ================================================================

def create_run_directory(run_id: str) -> str:
    """Create directory structure for run outputs (clears existing)."""
    run_dir = os.path.join(RESULTS_DIR, run_id)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    return run_dir


# ================================================================
# Per-year training loop
# ================================================================

def train_year(
    year: int,
    data_loader: DataLoader,
    feature_sets: Dict[int, List[str]],
    run_dir: str,
    horizon: int,
    group: str,
) -> Tuple[List[Dict], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Train RF_SHAP_5 and RF_SHAP_10 for a single year.

    Returns:
        (results, y_true_dict, y_pred_dict) keyed by model name.
    """
    print(f"\n{'='*60}")
    print(f"Year {year}")
    print(f"{'='*60}")

    train_df, test_df = data_loader.load_year(year)

    results: List[Dict] = []
    y_true_dict: Dict[str, np.ndarray] = {}
    y_pred_dict: Dict[str, np.ndarray] = {}

    for k, features in feature_sets.items():
        model_name = f"RF_SHAP_{k}"
        print(f"\n  {model_name} ({k} features: {', '.join(features)})")

        X_train, y_train, countries_train = extract_Xy(train_df, feature_cols=features)
        X_test, y_test, countries_test = extract_Xy(test_df, feature_cols=features)

        print(f"    Train: {len(y_train)} samples, {np.mean(y_train)*100:.1f}% crisis")
        print(f"    Test:  {len(y_test)} samples, {np.mean(y_test)*100:.1f}% crisis")

        model = RandomForestModel(
            name=model_name,
            n_estimators=RF_CONFIG["n_estimators"],
            mtry_grid=RF_CONFIG["mtry_grid"],
            cv_folds=RF_CONFIG["cv_folds"],
            class_weight=RF_CONFIG["class_weight"],
            random_state=RANDOM_SEED,
            n_jobs=RF_CONFIG["n_jobs"],
        )

        model.fit(X_train, y_train, countries=countries_train)
        preds = model.predict_proba(X_test)

        result = evaluate_predictions(y_test, preds, model_name, year, horizon, group)
        results.append(result)
        y_true_dict[model_name] = y_test
        y_pred_dict[model_name] = preds

        auc = result.get("AUC", np.nan)
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        print(f"    AUC: {auc_str} | best_mtry: {model.best_mtry}")

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

        # Save model
        model_dir = os.path.join(run_dir, "models", model_name)
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, f"model_t{year}"))

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
    results = []
    for model_name in all_y_true:
        result = evaluate_pooled(
            all_y_true[model_name], all_y_pred[model_name],
            model_name, horizon, group,
        )
        results.append(result)
    return results


# ================================================================
# Markdown summary
# ================================================================

def generate_summary_markdown(
    run_dir: str,
    config: Dict,
    feature_sets: Dict[int, List[str]],
    results_by_year: pd.DataFrame,
    results_overall: pd.DataFrame,
    runtime_seconds: float,
) -> str:
    md = []

    md.append("# SHAP Model Run Summary\n")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"**Runtime:** {runtime_seconds:.1f} seconds\n")

    md.append("\n## Configuration\n")
    md.append(f"- **Dataset:** expanded (SHAP features derived from expanded)")
    md.append(f"- **Horizon:** h={config['horizon']} years")
    md.append(f"- **Group:** {config['group']}")
    md.append(f"- **Years:** {config['years'][0]} - {config['years'][-1]} ({len(config['years'])} years)")
    md.append(f"- **CV Folds:** {N_FOLDS} (country-based)")
    md.append(f"- **Random Seed:** {RANDOM_SEED}")
    md.append(f"- **SHAP Source Model:** {SHAP_SOURCE_MODEL}\n")

    md.append("\n## Selected Features\n")
    for k, features in sorted(feature_sets.items()):
        md.append(f"\n### Top-{k} Features (RF_SHAP_{k})\n")
        md.append("| Rank | Feature |")
        md.append("|------|---------|")
        for i, feat in enumerate(features, 1):
            md.append(f"| {i} | {feat} |")
    md.append("")

    md.append("\n## Models\n")
    md.append("| Model | Features | RF Config |")
    md.append("|-------|----------|-----------|")
    for k in SHAP_K_VALUES:
        md.append(
            f"| RF_SHAP_{k} | Top-{k} SHAP features | "
            f"{RF_CONFIG['n_estimators']} trees, mtry grid {RF_CONFIG['mtry_grid']}, "
            f"{RF_CONFIG['cv_folds']}-fold CV |"
        )
    md.append("")

    md.append("\n## Overall Results (Pooled)\n")
    md.append("| Model | AUC | MSE | LogLik | Precision | Recall | F1 |")
    md.append("|-------|-----|-----|--------|-----------|--------|-----|")
    for _, row in results_overall.iterrows():
        vals = []
        for col in ["AUC", "MSE", "LogLik", "Precision", "Recall", "F1"]:
            v = row.get(col, np.nan)
            vals.append(f"{v:.4f}" if not pd.isna(v) else "N/A")
        md.append(f"| {row['Model']} | {' | '.join(vals)} |")
    md.append("")

    md.append("\n## Per-Year AUC\n")
    auc_pivot = results_by_year.pivot(index="Year", columns="Model", values="AUC")
    desired_order = [m for m in MODEL_NAMES if m in auc_pivot.columns]
    auc_pivot = auc_pivot[desired_order]
    model_cols = auc_pivot.columns.tolist()
    md.append("| Year | " + " | ".join(model_cols) + " |")
    md.append("|------|" + "|".join(["------"] * len(model_cols)) + "|")
    for year_val, row in auc_pivot.iterrows():
        vals = [f"{v:.4f}" if not pd.isna(v) else "N/A" for v in row.values]
        md.append(f"| {year_val} | " + " | ".join(vals) + " |")
    md.append("")

    md.append("\n## Output Files\n")
    md.append("- Results by year: `results_by_year.csv`")
    md.append("- Overall results: `results_overall.csv`")
    md.append("- Predictions: `predictions/{Model}/`")
    md.append("- Models: `models/{Model}/`")
    md.append("- Selected features: `selected_features.json`")
    md.append("")

    return "\n".join(md)


# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()

    run_id = args.run_id or f"shap_h{args.horizon}_{args.group}"

    print("=" * 60)
    print("SHAP MODEL (RF on SHAP-Selected Features)")
    print("=" * 60)
    print(f"Dataset:  expanded")
    print(f"Horizon:  h={args.horizon}")
    print(f"Group:    {args.group}")
    print(f"Models:   {', '.join(MODEL_NAMES)}")
    print(f"Run ID:   {run_id}")
    print("=" * 60)

    start_time = time.time()

    # Load SHAP rankings
    shap_df = load_shap_features(args.horizon, args.group)
    print(f"\nSHAP importance loaded from: {SHAP_SOURCE_MODEL}")
    print(f"Total features ranked: {len(shap_df)}")

    # Build feature sets for each K
    feature_sets: Dict[int, List[str]] = {}
    for k in SHAP_K_VALUES:
        features = get_top_k_features(shap_df, k)
        feature_sets[k] = features
        print(f"\n  Top-{k}: {', '.join(features)}")

    # Create run directory
    run_dir = create_run_directory(run_id)
    print(f"\nOutput directory: {run_dir}")

    # Initialize data loader (always expanded)
    data_loader = DataLoader("expanded", args.horizon, args.group)
    available_years = data_loader.available_years

    print(f"Available years: {available_years[0]} - {available_years[-1]} ({len(available_years)} years)")

    # Save config and selected features
    config = {
        "dataset": "expanded",
        "horizon": args.horizon,
        "group": args.group,
        "run_id": run_id,
        "years": available_years,
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
        "shap_source_model": SHAP_SOURCE_MODEL,
        "shap_k_values": SHAP_K_VALUES,
        "models": MODEL_NAMES,
        "rf_config": {
            "n_estimators": RF_CONFIG["n_estimators"],
            "mtry_grid": RF_CONFIG["mtry_grid"],
            "cv_folds": RF_CONFIG["cv_folds"],
            "class_weight": RF_CONFIG["class_weight"],
        },
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    selected_features = {
        f"top_{k}": features for k, features in feature_sets.items()
    }
    with open(os.path.join(run_dir, "selected_features.json"), "w") as f:
        json.dump(selected_features, f, indent=2)

    # Train and evaluate for each year
    all_results: List[Dict] = []
    all_y_true: Dict[str, List[np.ndarray]] = {}
    all_y_pred: Dict[str, List[np.ndarray]] = {}

    for year in available_years:
        year_results, y_true_dict, y_pred_dict = train_year(
            year, data_loader, feature_sets, run_dir, args.horizon, args.group,
        )
        all_results.extend(year_results)

        for model_name in y_true_dict:
            if model_name not in all_y_true:
                all_y_true[model_name] = []
                all_y_pred[model_name] = []
            all_y_true[model_name].append(y_true_dict[model_name])
            all_y_pred[model_name].append(y_pred_dict[model_name])

    # Pooled prediction files
    for model_name in all_y_true:
        pred_dfs = []
        for year in available_years:
            path = os.path.join(
                run_dir, "predictions", model_name, f"preds_t{year}.csv"
            )
            if os.path.exists(path):
                pred_dfs.append(pd.read_csv(path))
        if pred_dfs:
            pooled = pd.concat(pred_dfs, ignore_index=True)
            pooled.to_csv(
                os.path.join(run_dir, "predictions", model_name, "preds_pooled.csv"),
                index=False,
            )

    # Save per-year results
    results_by_year = pd.DataFrame(all_results)
    results_by_year.to_csv(os.path.join(run_dir, "results_by_year.csv"), index=False)

    # Compute and save pooled results
    pooled_results = compute_pooled_results(
        all_y_true, all_y_pred, args.horizon, args.group,
    )
    results_overall = pd.DataFrame(pooled_results)
    results_overall.to_csv(os.path.join(run_dir, "results_overall.csv"), index=False)

    # Runtime
    runtime_seconds = time.time() - start_time

    # Generate and save summary markdown
    summary_md = generate_summary_markdown(
        run_dir, config, feature_sets,
        results_by_year, results_overall, runtime_seconds,
    )
    with open(os.path.join(run_dir, "run_summary.md"), "w") as f:
        f.write(summary_md)

    # Print final summary
    print("\n" + "=" * 60)
    print("OVERALL RESULTS (Pooled)")
    print("=" * 60)
    print(f"{'Model':15s} | {'AUC':8s} | {'MSE':8s} | {'F1':8s}")
    print("-" * 50)

    results_overall["_sort"] = results_overall["Model"].apply(
        lambda x: MODEL_NAMES.index(x) if x in MODEL_NAMES else 99
    )
    results_overall = results_overall.sort_values("_sort").drop("_sort", axis=1)

    for _, row in results_overall.iterrows():
        auc = f"{row['AUC']:.4f}" if not pd.isna(row["AUC"]) else "N/A"
        mse = f"{row['MSE']:.4f}" if not pd.isna(row["MSE"]) else "N/A"
        f1 = f"{row['F1']:.4f}" if not pd.isna(row["F1"]) else "N/A"
        print(f"{row['Model']:15s} | {auc:8s} | {mse:8s} | {f1:8s}")

    print("\n" + "=" * 60)
    print(f"Run completed in {runtime_seconds:.1f} seconds")
    print(f"Results saved to: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
