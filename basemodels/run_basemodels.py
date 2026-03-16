# ================================================================
# Base Models - Main Runner
# ================================================================
#
# Trains and evaluates all base models on a given dataset/horizon/group.
#
# Usage:
#   python -m basemodels.run_basemodels --dataset baseline --horizon 2 --group ALL
#
# Or from the project root:
#   python basemodels/run_basemodels.py --dataset baseline --horizon 2 --group ALL
#
# ================================================================

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
import functools

# Force unbuffered output
print = functools.partial(print, flush=True)

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from basemodels.config import (
    VALID_DATASETS,
    VALID_HORIZONS,
    VALID_GROUPS,
    RESULTS_DIR,
    N_FOLDS,
    RANDOM_SEED,
    RF_CONFIG,
    ADABOOST_CONFIG,
    XGB_CONFIG,
    PROBIT_CONFIG,
)
from basemodels.data_loader import DataLoader
from basemodels.models import (
    RuleOfThumbModel,
    ProbitModel,
    RandomForestModel,
    AdaBoostModel,
    XGBoostModel,
)
from metrics.evaluation import evaluate_predictions, evaluate_pooled

# Suppress warnings
warnings.filterwarnings("ignore")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Base Models for sovereign debt crisis prediction"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=VALID_DATASETS,
        required=True,
        help="Dataset: 'baseline' or 'expanded'",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=VALID_HORIZONS,
        required=True,
        help="Forecast horizon: 2, 5, or 10 years",
    )
    parser.add_argument(
        "--group",
        type=str,
        choices=VALID_GROUPS,
        required=True,
        help="Country group: 'ALL', 'EME', or 'LIC'",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID (default: auto-generated timestamp)",
    )
    return parser.parse_args()


def create_run_directory(run_id: str) -> str:
    """Create directory structure for run outputs.
    
    Deletes any existing directory with the same run_id first so that
    results for a given (dataset, horizon, group) are always fresh.
    """
    import shutil
    run_dir = os.path.join(RESULTS_DIR, run_id)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    return run_dir


def create_models(horizon: int) -> List:
    """
    Create instances of all base models.
    
    Args:
        horizon: Forecast horizon (needed for Rule of Thumb)
    
    Returns:
        List of model instances
    """
    return [
        RuleOfThumbModel(name="RuleOfThumb", horizon=horizon, random_state=RANDOM_SEED),
        ProbitModel(name="Probit", random_state=RANDOM_SEED),
        RandomForestModel(name="RandomForest", random_state=RANDOM_SEED),
        AdaBoostModel(name="AdaBoost", random_state=RANDOM_SEED),
        XGBoostModel(name="XGBoost", random_state=RANDOM_SEED),
    ]


def train_year(
    year: int,
    data_loader: DataLoader,
    run_dir: str,
    horizon: int,
    group: str,
) -> Tuple[List[Dict], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Train all models for a single year and return results.
    
    Args:
        year: Test year
        data_loader: DataLoader instance
        run_dir: Run output directory
        horizon: Forecast horizon
        group: Country group
    
    Returns:
        Tuple of (results_list, y_true_dict, y_pred_dict)
    """
    print(f"\n{'='*60}")
    print(f"Year {year}")
    print(f"{'='*60}")
    
    # Load data with CH column for Rule of Thumb
    X_train, y_train, countries_train, ch_train, X_test, y_test, countries_test, ch_test = \
        data_loader.load_year_Xy_with_ch(year)
    
    print(f"  Train: {len(y_train)} samples, {np.mean(y_train)*100:.1f}% crisis")
    print(f"  Test:  {len(y_test)} samples, {np.mean(y_test)*100:.1f}% crisis")
    
    # Create models
    models = create_models(horizon)
    
    # Train and evaluate each model
    results = []
    y_true_dict = {}
    y_pred_dict = {}
    
    for model in models:
        print(f"\n  Training {model.name}...")
        
        # Prepare validation set for early stopping (XGBoost)
        val_size = int(0.2 * len(y_train))
        idx = np.random.RandomState(RANDOM_SEED).permutation(len(y_train))
        train_idx, val_idx = idx[val_size:], idx[:val_size]
        
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        countries_tr = countries_train[train_idx]
        
        # Fit model with appropriate arguments
        if model.name == "RuleOfThumb":
            # Rule of Thumb doesn't need training
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_test, ch=ch_test)
        elif model.name in ["RandomForest", "AdaBoost"]:
            # These models need countries for CV-based tuning
            model.fit(X_tr, y_tr, countries=countries_tr)
            preds = model.predict_proba(X_test)
        elif model.name == "XGBoost":
            # XGBoost uses validation set for early stopping
            model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
            preds = model.predict_proba(X_test)
        else:
            # Probit - simple fit
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_test)
        
        # Evaluate
        result = evaluate_predictions(y_test, preds, model.name, year, horizon, group)
        results.append(result)
        
        y_true_dict[model.name] = y_test
        y_pred_dict[model.name] = preds
        
        # Save predictions (with identifiers for alignment in significance tests)
        pred_dir = os.path.join(run_dir, "predictions", model.name)
        os.makedirs(pred_dir, exist_ok=True)
        pred_df = pd.DataFrame({
            "Year": np.full(len(y_test), year, dtype=np.int64),
            "WEOCountryCode": countries_test,
            "y_true": y_test,
            "y_pred": preds,
        })
        pred_df.to_csv(os.path.join(pred_dir, f"preds_t{year}.csv"), index=False)
        
        # Save model
        model_dir = os.path.join(run_dir, "models", model.name)
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, f"model_t{year}"))
        
        # Print result
        auc = result.get("AUC", np.nan)
        if model.name in ["RandomForest"]:
            print(f"    {model.name:15s} | AUC: {auc:.4f} | best_mtry: {model.best_mtry}")
        elif model.name in ["AdaBoost"]:
            print(f"    {model.name:15s} | AUC: {auc:.4f} | best_n_estimators: {model.best_n_estimators}")
        elif model.name in ["XGBoost"]:
            print(f"    {model.name:15s} | AUC: {auc:.4f} | best_iteration: {model.best_iteration}")
        else:
            print(f"    {model.name:15s} | AUC: {auc:.4f}" if not np.isnan(auc) else f"    {model.name:15s} | AUC: N/A")
    
    return results, y_true_dict, y_pred_dict


def compute_pooled_results(
    all_y_true: Dict[str, List[np.ndarray]],
    all_y_pred: Dict[str, List[np.ndarray]],
    horizon: int,
    group: str,
) -> List[Dict]:
    """
    Compute pooled (overall) evaluation metrics.
    
    Args:
        all_y_true: Dict mapping model name to list of y_true arrays per year
        all_y_pred: Dict mapping model name to list of y_pred arrays per year
        horizon: Forecast horizon
        group: Country group
    
    Returns:
        List of pooled evaluation result dicts
    """
    results = []
    
    for model_name in all_y_true.keys():
        y_list = all_y_true[model_name]
        p_list = all_y_pred[model_name]
        
        result = evaluate_pooled(y_list, p_list, model_name, horizon, group)
        results.append(result)
    
    return results


def generate_summary_markdown(
    run_dir: str,
    config: Dict,
    results_by_year: pd.DataFrame,
    results_overall: pd.DataFrame,
    runtime_seconds: float,
) -> str:
    """
    Generate a markdown summary of the run.
    
    Args:
        run_dir: Run output directory
        config: Run configuration
        results_by_year: Per-year results DataFrame
        results_overall: Overall results DataFrame
        runtime_seconds: Total runtime in seconds
    
    Returns:
        Markdown content as string
    """
    md_lines = []
    
    # Header
    md_lines.append("# Base Models Run Summary\n")
    md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_lines.append(f"**Runtime:** {runtime_seconds:.1f} seconds\n")
    
    # Configuration
    md_lines.append("\n## Configuration\n")
    md_lines.append(f"- **Dataset:** {config['dataset']}")
    md_lines.append(f"- **Horizon:** h={config['horizon']} years")
    md_lines.append(f"- **Group:** {config['group']}")
    md_lines.append(f"- **Years:** {config['years'][0]} - {config['years'][-1]} ({len(config['years'])} years)")
    md_lines.append(f"- **CV Folds:** {N_FOLDS} (country-based)")
    md_lines.append(f"- **Random Seed:** {RANDOM_SEED}\n")
    
    # Models
    md_lines.append("\n## Models\n")
    md_lines.append("| Model | Description |")
    md_lines.append("|-------|-------------|")
    md_lines.append(f"| Rule of Thumb | P = 1 - (1 - CH/100)^h, baseline using historical crisis rate |")
    md_lines.append(f"| Probit | GLM with probit link (statsmodels) |")
    md_lines.append(f"| Random Forest | {RF_CONFIG['n_estimators']} trees, mtry grid {RF_CONFIG['mtry_grid']} via {RF_CONFIG['cv_folds']}-fold CV |")
    md_lines.append(f"| AdaBoost | GBM with shrinkage={ADABOOST_CONFIG['learning_rate']}, depth={ADABOOST_CONFIG['max_depth']}, n_trees grid {ADABOOST_CONFIG['n_estimators_grid']} |")
    md_lines.append(f"| XGBoost | {XGB_CONFIG['n_estimators']} rounds, early stopping, lr={XGB_CONFIG['learning_rate']} |")
    md_lines.append("")
    
    # Overall Results
    md_lines.append("\n## Overall Results (Pooled)\n")
    md_lines.append("| Model | AUC | MSE | LogLik | Precision | Recall | F1 |")
    md_lines.append("|-------|-----|-----|--------|-----------|--------|-----|")
    
    for _, row in results_overall.iterrows():
        auc = f"{row['AUC']:.4f}" if not pd.isna(row['AUC']) else "N/A"
        mse = f"{row['MSE']:.4f}" if not pd.isna(row['MSE']) else "N/A"
        ll = f"{row['LogLik']:.4f}" if not pd.isna(row['LogLik']) else "N/A"
        prec = f"{row['Precision']:.4f}" if not pd.isna(row['Precision']) else "N/A"
        rec = f"{row['Recall']:.4f}" if not pd.isna(row['Recall']) else "N/A"
        f1 = f"{row['F1']:.4f}" if not pd.isna(row['F1']) else "N/A"
        md_lines.append(f"| {row['Model']} | {auc} | {mse} | {ll} | {prec} | {rec} | {f1} |")
    
    md_lines.append("")
    
    # Per-Year Results (AUC only for brevity)
    md_lines.append("\n## Per-Year AUC\n")
    
    # Pivot table for AUC by year and model
    auc_pivot = results_by_year.pivot(index="Year", columns="Model", values="AUC")
    
    # Reorder columns to match R output order
    desired_order = ["RuleOfThumb", "Probit", "RandomForest", "AdaBoost", "XGBoost"]
    available_cols = [c for c in desired_order if c in auc_pivot.columns]
    auc_pivot = auc_pivot[available_cols]
    
    # Create header
    model_cols = auc_pivot.columns.tolist()
    header = "| Year | " + " | ".join(model_cols) + " |"
    sep = "|------|" + "|".join(["------"] * len(model_cols)) + "|"
    md_lines.append(header)
    md_lines.append(sep)
    
    for year, row in auc_pivot.iterrows():
        vals = [f"{v:.4f}" if not pd.isna(v) else "N/A" for v in row.values]
        md_lines.append(f"| {year} | " + " | ".join(vals) + " |")
    
    md_lines.append("")
    
    # File locations
    md_lines.append("\n## Output Files\n")
    md_lines.append(f"- Results by year: `{os.path.join(run_dir, 'results_by_year.csv')}`")
    md_lines.append(f"- Overall results: `{os.path.join(run_dir, 'results_overall.csv')}`")
    md_lines.append(f"- Predictions: `{os.path.join(run_dir, 'predictions/')}`")
    md_lines.append(f"- Models: `{os.path.join(run_dir, 'models/')}`")
    md_lines.append("")
    
    return "\n".join(md_lines)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Generate run ID (fixed per dataset/horizon/group -- overwrites previous run)
    run_id = args.run_id or f"{args.dataset}_h{args.horizon}_{args.group}"
    
    print("=" * 60)
    print("BASE MODELS")
    print("=" * 60)
    print(f"Dataset:  {args.dataset}")
    print(f"Horizon:  h={args.horizon}")
    print(f"Group:    {args.group}")
    print(f"Run ID:   {run_id}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create run directory
    run_dir = create_run_directory(run_id)
    print(f"\nOutput directory: {run_dir}")
    
    # Initialize data loader
    data_loader = DataLoader(args.dataset, args.horizon, args.group)
    available_years = data_loader.available_years
    
    print(f"Available years: {available_years[0]} - {available_years[-1]} ({len(available_years)} years)")
    print(f"Features: {len(data_loader.feature_cols)} columns")
    
    # Save config
    config = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "group": args.group,
        "run_id": run_id,
        "years": available_years,
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
        "feature_cols": data_loader.feature_cols,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Train and evaluate for each year
    all_results = []
    all_y_true = {}  # model_name -> list of y_true arrays
    all_y_pred = {}  # model_name -> list of y_pred arrays
    
    for year in available_years:
        year_results, y_true_dict, y_pred_dict = train_year(
            year, data_loader, run_dir, args.horizon, args.group
        )
        all_results.extend(year_results)
        
        # Accumulate for pooled evaluation
        for model_name in y_true_dict.keys():
            if model_name not in all_y_true:
                all_y_true[model_name] = []
                all_y_pred[model_name] = []
            all_y_true[model_name].append(y_true_dict[model_name])
            all_y_pred[model_name].append(y_pred_dict[model_name])
    
    # One pooled prediction file per model (for significance tests without rerunning)
    for model_name in all_y_true.keys():
        pred_dfs = []
        for year in available_years:
            path = os.path.join(run_dir, "predictions", model_name, f"preds_t{year}.csv")
            if os.path.exists(path):
                pred_dfs.append(pd.read_csv(path))
        if pred_dfs:
            pooled = pd.concat(pred_dfs, ignore_index=True)
            pooled_path = os.path.join(run_dir, "predictions", model_name, "preds_pooled.csv")
            pooled.to_csv(pooled_path, index=False)
    
    # Save per-year results
    results_by_year = pd.DataFrame(all_results)
    results_by_year.to_csv(os.path.join(run_dir, "results_by_year.csv"), index=False)
    
    # Compute and save pooled results
    pooled_results = compute_pooled_results(all_y_true, all_y_pred, args.horizon, args.group)
    results_overall = pd.DataFrame(pooled_results)
    results_overall.to_csv(os.path.join(run_dir, "results_overall.csv"), index=False)
    
    # Calculate runtime
    runtime_seconds = time.time() - start_time
    
    # Generate and save summary markdown
    summary_md = generate_summary_markdown(
        run_dir, config, results_by_year, results_overall, runtime_seconds
    )
    with open(os.path.join(run_dir, "run_summary.md"), "w") as f:
        f.write(summary_md)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("OVERALL RESULTS (Pooled)")
    print("=" * 60)
    print(f"{'Model':15s} | {'AUC':8s} | {'MSE':8s} | {'F1':8s}")
    print("-" * 50)
    
    # Sort by model order
    model_order = ["RuleOfThumb", "Probit", "RandomForest", "AdaBoost", "XGBoost"]
    results_overall["_sort"] = results_overall["Model"].apply(
        lambda x: model_order.index(x) if x in model_order else 99
    )
    results_overall = results_overall.sort_values("_sort").drop("_sort", axis=1)
    
    for _, row in results_overall.iterrows():
        auc = f"{row['AUC']:.4f}" if not pd.isna(row['AUC']) else "N/A"
        mse = f"{row['MSE']:.4f}" if not pd.isna(row['MSE']) else "N/A"
        f1 = f"{row['F1']:.4f}" if not pd.isna(row['F1']) else "N/A"
        print(f"{row['Model']:15s} | {auc:8s} | {mse:8s} | {f1:8s}")
    
    print("\n" + "=" * 60)
    print(f"Run completed in {runtime_seconds:.1f} seconds")
    print(f"Results saved to: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
