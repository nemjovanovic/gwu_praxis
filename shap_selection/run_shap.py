# ================================================================
# SHAP Selection - Main Runner
# ================================================================
#
# Trains tree-based models (RF, AdaBoost, XGBoost) on the expanded
# dataset, computes SHAP values via TreeExplainer, and persists all
# artifacts for downstream analysis and charting.
#
# Usage (from the project root):
#   python shap_selection/run_shap.py --horizon 2 --group ALL
#   python shap_selection/run_shap.py --horizon 5 --group EME
#   python shap_selection/run_shap.py --horizon 10 --group LIC
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

# Force unbuffered output
print = functools.partial(print, flush=True)

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import shap

from basemodels.config import (
    VALID_HORIZONS,
    VALID_GROUPS,
    N_FOLDS,
    RANDOM_SEED,
    RF_CONFIG,
    ADABOOST_CONFIG,
    XGB_CONFIG,
)
from basemodels.data_loader import DataLoader
from basemodels.models import (
    RandomForestModel,
    AdaBoostModel,
    XGBoostModel,
)
from metrics.evaluation import evaluate_predictions, evaluate_pooled

from shap_selection.config import (
    SHAP_RESULTS_DIR,
    MAX_SHAP_SAMPLES,
    SHAP_MODELS,
)

# Suppress warnings
warnings.filterwarnings("ignore")


# ================================================================
# CLI
# ================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute SHAP values for tree-based models on the expanded dataset"
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
        help="Custom run ID (default: expanded_h{horizon}_{group})",
    )
    return parser.parse_args()


# ================================================================
# Directory setup
# ================================================================

def create_run_directory(run_id: str) -> str:
    """Create directory structure for run outputs.

    Deletes any existing directory with the same run_id first so that
    results are always fresh.
    """
    run_dir = os.path.join(SHAP_RESULTS_DIR, run_id)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "shap"), exist_ok=True)
    return run_dir


# ================================================================
# Model creation
# ================================================================

def create_models() -> List:
    """Create instances of the three tree-based models."""
    return [
        RandomForestModel(name="RandomForest", random_state=RANDOM_SEED),
        AdaBoostModel(name="AdaBoost", random_state=RANDOM_SEED),
        XGBoostModel(name="XGBoost", random_state=RANDOM_SEED),
    ]


# ================================================================
# SHAP computation
# ================================================================

def compute_shap_values(
    model,
    X_train: np.ndarray,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SHAP values for a trained tree-based model.

    Args:
        model: A fitted model wrapper whose `.model` attribute is the
               underlying sklearn/xgboost estimator.
        X_train: Full training feature matrix.
        feature_cols: Feature column names.

    Returns:
        (sv_signed, X_shap, mean_abs_shap)
            sv_signed:     SHAP values for the positive class, shape (n, p).
            X_shap:        The (sub)sample of X_train used for SHAP.
            mean_abs_shap: Mean |SHAP| per feature, shape (p,).
    """
    # Subsample if needed
    if X_train.shape[0] > MAX_SHAP_SAMPLES:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(X_train.shape[0], MAX_SHAP_SAMPLES, replace=False)
        X_shap = X_train[idx]
    else:
        X_shap = X_train

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_shap)

    # Handle multi-output shapes:
    #   RF  -> list of [neg_class, pos_class] arrays
    #   GBM -> 2D array (single output)
    #   XGB -> may be list or 2D depending on version
    if isinstance(shap_values, list):
        sv_signed = shap_values[1]
    elif shap_values.ndim == 3:
        sv_signed = shap_values[:, :, 1]
    else:
        sv_signed = shap_values

    sv_signed = np.asarray(sv_signed, dtype=np.float32)
    mean_abs_shap = np.abs(sv_signed).mean(axis=0)

    return sv_signed, X_shap, mean_abs_shap


def save_shap_artifacts(
    sv_signed: np.ndarray,
    X_shap: np.ndarray,
    mean_abs_shap: np.ndarray,
    feature_cols: List[str],
    save_dir: str,
    year: int,
) -> None:
    """Persist SHAP artifacts for a single model and year."""
    os.makedirs(save_dir, exist_ok=True)

    # Compressed numpy archive (for beeswarm / force plots later)
    np.savez_compressed(
        os.path.join(save_dir, f"shap_values_t{year}.npz"),
        shap_values=sv_signed,
        X_shap=X_shap.astype(np.float32),
        feature_names=np.array(feature_cols, dtype=object),
    )

    # Human-readable importance CSV
    importance_df = (
        pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap,
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df.to_csv(
        os.path.join(save_dir, f"shap_importance_t{year}.csv"), index=False
    )


# ================================================================
# Per-year training + SHAP loop
# ================================================================

def train_year(
    year: int,
    data_loader: DataLoader,
    run_dir: str,
    horizon: int,
    group: str,
    feature_cols: List[str],
) -> Tuple[List[Dict], Dict[str, np.ndarray], Dict[str, np.ndarray],
           Dict[str, np.ndarray]]:
    """
    Train all tree models for a single year, compute SHAP, save artifacts.

    Returns:
        (results, y_true_dict, y_pred_dict, mean_abs_shap_dict)
    """
    print(f"\n{'='*60}")
    print(f"Year {year}")
    print(f"{'='*60}")

    X_train, y_train, countries_train, _ch_train, \
        X_test, y_test, countries_test, _ch_test = \
        data_loader.load_year_Xy_with_ch(year)

    print(f"  Train: {len(y_train)} samples, {np.mean(y_train)*100:.1f}% crisis")
    print(f"  Test:  {len(y_test)} samples, {np.mean(y_test)*100:.1f}% crisis")

    models = create_models()

    results: List[Dict] = []
    y_true_dict: Dict[str, np.ndarray] = {}
    y_pred_dict: Dict[str, np.ndarray] = {}
    mean_abs_shap_dict: Dict[str, np.ndarray] = {}

    # Validation split (for XGBoost early stopping)
    val_size = int(0.2 * len(y_train))
    idx = np.random.RandomState(RANDOM_SEED).permutation(len(y_train))
    train_idx, val_idx = idx[val_size:], idx[:val_size]

    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    countries_tr = countries_train[train_idx]

    for model in models:
        print(f"\n  Training {model.name}...")

        # ----- Fit -----
        if model.name in ["RandomForest", "AdaBoost"]:
            model.fit(X_tr, y_tr, countries=countries_tr)
        elif model.name == "XGBoost":
            model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_tr, y_tr)

        # ----- Predict -----
        preds = model.predict_proba(X_test)

        # ----- Evaluate -----
        result = evaluate_predictions(y_test, preds, model.name, year, horizon, group)
        results.append(result)
        y_true_dict[model.name] = y_test
        y_pred_dict[model.name] = preds

        # ----- Save predictions -----
        pred_dir = os.path.join(run_dir, "predictions", model.name)
        os.makedirs(pred_dir, exist_ok=True)
        pred_df = pd.DataFrame({
            "Year": np.full(len(y_test), year, dtype=np.int64),
            "WEOCountryCode": countries_test,
            "y_true": y_test,
            "y_pred": preds,
        })
        pred_df.to_csv(os.path.join(pred_dir, f"preds_t{year}.csv"), index=False)

        # ----- Save model -----
        model_dir = os.path.join(run_dir, "models", model.name)
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, f"model_t{year}"))

        # ----- Compute SHAP -----
        print(f"    Computing SHAP values...")
        sv_signed, X_shap, mean_abs_shap = compute_shap_values(
            model, X_train, feature_cols
        )
        mean_abs_shap_dict[model.name] = mean_abs_shap

        shap_dir = os.path.join(run_dir, "shap", model.name)
        save_shap_artifacts(sv_signed, X_shap, mean_abs_shap, feature_cols,
                            shap_dir, year)

        # ----- Print summary line -----
        auc = result.get("AUC", np.nan)
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        top3 = np.argsort(mean_abs_shap)[::-1][:3]
        top3_names = [feature_cols[i] for i in top3]
        extra = ""
        if model.name == "RandomForest":
            extra = f" | best_mtry: {model.best_mtry}"
        elif model.name == "AdaBoost":
            extra = f" | best_n_estimators: {model.best_n_estimators}"
        elif model.name == "XGBoost":
            extra = f" | best_iteration: {model.best_iteration}"
        print(f"    {model.name:15s} | AUC: {auc_str}{extra}")
        print(f"    {'':15s} | SHAP top-3: {', '.join(top3_names)}")

    return results, y_true_dict, y_pred_dict, mean_abs_shap_dict


# ================================================================
# Pooled evaluation
# ================================================================

def compute_pooled_results(
    all_y_true: Dict[str, List[np.ndarray]],
    all_y_pred: Dict[str, List[np.ndarray]],
    horizon: int,
    group: str,
) -> List[Dict]:
    """Compute pooled (overall) evaluation metrics."""
    results = []
    for model_name in all_y_true:
        result = evaluate_pooled(
            all_y_true[model_name], all_y_pred[model_name],
            model_name, horizon, group,
        )
        results.append(result)
    return results


# ================================================================
# Pooled SHAP importance
# ================================================================

def save_pooled_shap_importance(
    all_mean_abs_shap: Dict[str, List[np.ndarray]],
    feature_cols: List[str],
    run_dir: str,
) -> None:
    """Average mean_abs_shap across years and save per model."""
    for model_name, shap_list in all_mean_abs_shap.items():
        stacked = np.vstack(shap_list)               # (n_years, n_features)
        pooled_mean = stacked.mean(axis=0)            # (n_features,)

        importance_df = (
            pd.DataFrame({
                "feature": feature_cols,
                "mean_abs_shap": pooled_mean,
            })
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        importance_df["rank"] = range(1, len(importance_df) + 1)

        out_dir = os.path.join(run_dir, "shap", model_name)
        os.makedirs(out_dir, exist_ok=True)
        importance_df.to_csv(
            os.path.join(out_dir, "shap_importance_pooled.csv"), index=False
        )


# ================================================================
# Markdown summary
# ================================================================

def generate_summary_markdown(
    run_dir: str,
    config: Dict,
    results_by_year: pd.DataFrame,
    results_overall: pd.DataFrame,
    runtime_seconds: float,
) -> str:
    """Generate a markdown summary of the run."""
    md = []

    md.append("# SHAP Selection Run Summary\n")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"**Runtime:** {runtime_seconds:.1f} seconds\n")

    # Configuration
    md.append("\n## Configuration\n")
    md.append(f"- **Dataset:** expanded (always)")
    md.append(f"- **Horizon:** h={config['horizon']} years")
    md.append(f"- **Group:** {config['group']}")
    md.append(f"- **Years:** {config['years'][0]} - {config['years'][-1]} ({len(config['years'])} years)")
    md.append(f"- **CV Folds:** {N_FOLDS} (country-based)")
    md.append(f"- **Random Seed:** {RANDOM_SEED}")
    md.append(f"- **SHAP subsample:** {MAX_SHAP_SAMPLES} rows per year\n")

    # Models
    md.append("\n## Models\n")
    md.append("| Model | Description |")
    md.append("|-------|-------------|")
    md.append(f"| Random Forest | {RF_CONFIG['n_estimators']} trees, mtry grid {RF_CONFIG['mtry_grid']} via {RF_CONFIG['cv_folds']}-fold CV |")
    md.append(f"| AdaBoost | GBM with shrinkage={ADABOOST_CONFIG['learning_rate']}, depth={ADABOOST_CONFIG['max_depth']}, n_trees grid {ADABOOST_CONFIG['n_estimators_grid']} |")
    md.append(f"| XGBoost | {XGB_CONFIG['n_estimators']} rounds, early stopping, lr={XGB_CONFIG['learning_rate']} |")
    md.append("")

    # Overall Results
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

    # Per-Year AUC
    md.append("\n## Per-Year AUC\n")
    auc_pivot = results_by_year.pivot(index="Year", columns="Model", values="AUC")
    desired_order = [m for m in SHAP_MODELS if m in auc_pivot.columns]
    auc_pivot = auc_pivot[desired_order]
    model_cols = auc_pivot.columns.tolist()
    md.append("| Year | " + " | ".join(model_cols) + " |")
    md.append("|------|" + "|".join(["------"] * len(model_cols)) + "|")
    for year, row in auc_pivot.iterrows():
        vals = [f"{v:.4f}" if not pd.isna(v) else "N/A" for v in row.values]
        md.append(f"| {year} | " + " | ".join(vals) + " |")
    md.append("")

    # SHAP top-10 (pooled) per model
    md.append("\n## SHAP Top-10 Features (Pooled Across Years)\n")
    for model_name in SHAP_MODELS:
        pooled_path = os.path.join(run_dir, "shap", model_name,
                                   "shap_importance_pooled.csv")
        if not os.path.isfile(pooled_path):
            continue
        imp = pd.read_csv(pooled_path)
        md.append(f"\n### {model_name}\n")
        md.append("| Rank | Feature | Mean |SHAP| |")
        md.append("|------|---------|--------------|")
        for _, r in imp.head(10).iterrows():
            md.append(f"| {int(r['rank'])} | {r['feature']} | {r['mean_abs_shap']:.6f} |")
        md.append("")

    # Output files
    md.append("\n## Output Files\n")
    md.append(f"- Results by year: `results_by_year.csv`")
    md.append(f"- Overall results: `results_overall.csv`")
    md.append(f"- Predictions: `predictions/{{Model}}/`")
    md.append(f"- SHAP artifacts: `shap/{{Model}}/`")
    md.append(f"- Models: `models/{{Model}}/`")
    md.append("")

    return "\n".join(md)


# ================================================================
# Main
# ================================================================

def main():
    """Main entry point."""
    args = parse_args()

    run_id = args.run_id or f"expanded_h{args.horizon}_{args.group}"

    print("=" * 60)
    print("SHAP SELECTION (Tree-Based Models + Shapley Values)")
    print("=" * 60)
    print(f"Dataset:  expanded")
    print(f"Horizon:  h={args.horizon}")
    print(f"Group:    {args.group}")
    print(f"Models:   {', '.join(SHAP_MODELS)}")
    print(f"Run ID:   {run_id}")
    print("=" * 60)

    start_time = time.time()

    # Create run directory
    run_dir = create_run_directory(run_id)
    print(f"\nOutput directory: {run_dir}")

    # Initialize data loader (always expanded)
    data_loader = DataLoader("expanded", args.horizon, args.group)
    available_years = data_loader.available_years
    feature_cols = data_loader.feature_cols

    print(f"Available years: {available_years[0]} - {available_years[-1]} ({len(available_years)} years)")
    print(f"Features: {len(feature_cols)} columns")

    # Save config
    config = {
        "dataset": "expanded",
        "horizon": args.horizon,
        "group": args.group,
        "run_id": run_id,
        "years": available_years,
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
        "max_shap_samples": MAX_SHAP_SAMPLES,
        "models": SHAP_MODELS,
        "feature_cols": feature_cols,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Train, evaluate, compute SHAP for each year
    all_results: List[Dict] = []
    all_y_true: Dict[str, List[np.ndarray]] = {}
    all_y_pred: Dict[str, List[np.ndarray]] = {}
    all_mean_abs_shap: Dict[str, List[np.ndarray]] = {}

    for year in available_years:
        year_results, y_true_dict, y_pred_dict, mean_abs_shap_dict = train_year(
            year, data_loader, run_dir, args.horizon, args.group, feature_cols
        )
        all_results.extend(year_results)

        for model_name in y_true_dict:
            if model_name not in all_y_true:
                all_y_true[model_name] = []
                all_y_pred[model_name] = []
                all_mean_abs_shap[model_name] = []
            all_y_true[model_name].append(y_true_dict[model_name])
            all_y_pred[model_name].append(y_pred_dict[model_name])
            all_mean_abs_shap[model_name].append(mean_abs_shap_dict[model_name])

    # Pooled prediction files
    for model_name in all_y_true:
        pred_dfs = []
        for year in available_years:
            path = os.path.join(run_dir, "predictions", model_name,
                                f"preds_t{year}.csv")
            if os.path.exists(path):
                pred_dfs.append(pd.read_csv(path))
        if pred_dfs:
            pooled = pd.concat(pred_dfs, ignore_index=True)
            pooled.to_csv(
                os.path.join(run_dir, "predictions", model_name,
                             "preds_pooled.csv"),
                index=False,
            )

    # Save per-year results
    results_by_year = pd.DataFrame(all_results)
    results_by_year.to_csv(os.path.join(run_dir, "results_by_year.csv"), index=False)

    # Compute and save pooled results
    pooled_results = compute_pooled_results(
        all_y_true, all_y_pred, args.horizon, args.group
    )
    results_overall = pd.DataFrame(pooled_results)
    results_overall.to_csv(os.path.join(run_dir, "results_overall.csv"), index=False)

    # Save pooled SHAP importance
    save_pooled_shap_importance(all_mean_abs_shap, feature_cols, run_dir)

    # Runtime
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

    model_order = SHAP_MODELS
    results_overall["_sort"] = results_overall["Model"].apply(
        lambda x: model_order.index(x) if x in model_order else 99
    )
    results_overall = results_overall.sort_values("_sort").drop("_sort", axis=1)

    for _, row in results_overall.iterrows():
        auc = f"{row['AUC']:.4f}" if not pd.isna(row['AUC']) else "N/A"
        mse = f"{row['MSE']:.4f}" if not pd.isna(row['MSE']) else "N/A"
        f1 = f"{row['F1']:.4f}" if not pd.isna(row['F1']) else "N/A"
        print(f"{row['Model']:15s} | {auc:8s} | {mse:8s} | {f1:8s}")

    # Print top-5 SHAP features per model
    print("\n" + "=" * 60)
    print("TOP-5 SHAP FEATURES (Pooled)")
    print("=" * 60)
    for model_name in SHAP_MODELS:
        pooled_path = os.path.join(run_dir, "shap", model_name,
                                   "shap_importance_pooled.csv")
        if not os.path.isfile(pooled_path):
            continue
        imp = pd.read_csv(pooled_path)
        print(f"\n  {model_name}:")
        for _, r in imp.head(5).iterrows():
            print(f"    {int(r['rank']):2d}. {r['feature']:30s} | mean|SHAP|={r['mean_abs_shap']:.6f}")

    print("\n" + "=" * 60)
    print(f"Run completed in {runtime_seconds:.1f} seconds")
    print(f"Results saved to: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
