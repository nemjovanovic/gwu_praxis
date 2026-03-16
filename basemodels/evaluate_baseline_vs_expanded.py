# ================================================================
# Baseline vs Expanded evaluator
# Pairs runs by (horizon, group), compares AUC (DeLong) and F1 (bootstrap)
# Writes one comparison table per (group, horizon).
# ================================================================

import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from basemodels.config import RESULTS_DIR, VALID_HORIZONS, VALID_GROUPS
from metrics.significance_tests import delong_test, bootstrap_f1_test

warnings.filterwarnings("ignore")

# Default model list (same as run_basemodels)
DEFAULT_MODELS = ["RuleOfThumb", "Probit", "RandomForest", "AdaBoost", "XGBoost"]
ALPHA = 0.05
N_BOOTSTRAP = 1000
RANDOM_STATE = 1234
OUTPUT_SUBDIR = "baseline_vs_expanded"

# Run dir pattern: {dataset}_h{horizon}_{group} (no timestamp)
RUN_PATTERN = re.compile(
    r"^(baseline|expanded)_h(\d+)_(ALL|EME|LIC)$"
)


def discover_runs(results_dir: str) -> Dict[Tuple[int, str], Dict[str, List[str]]]:
    """
    List subdirs of results_dir, parse (dataset, horizon, group), validate
    (config.json + predictions/<Model>/preds_pooled.csv for each model).
    Returns: {(horizon, group): {"baseline": [run_id, ...], "expanded": [run_id, ...]}}
    """
    out: Dict[Tuple[int, str], Dict[str, List[str]]] = {}
    if not os.path.isdir(results_dir):
        return out
    for name in os.listdir(results_dir):
        m = RUN_PATTERN.match(name)
        if not m:
            continue
        dataset, horizon_str, group = m.groups()
        horizon = int(horizon_str)
        run_path = os.path.join(results_dir, name)
        if not os.path.isdir(run_path):
            continue
        config_path = os.path.join(run_path, "config.json")
        if not os.path.isfile(config_path):
            continue
        # Check predictions per model (use default model list to know what we need)
        valid = True
        for model in DEFAULT_MODELS:
            pred_path = os.path.join(run_path, "predictions", model, "preds_pooled.csv")
            if not os.path.isfile(pred_path):
                valid = False
                break
        if not valid:
            continue
        key = (horizon, group)
        if key not in out:
            out[key] = {"baseline": [], "expanded": []}
        out[key][dataset].append(name)
    return out


def pick_per_dataset(
    runs: Dict[str, List[str]]
) -> Tuple[Optional[str], Optional[str]]:
    """Pick the baseline and expanded run directories (one per dataset)."""
    baseline_list = runs.get("baseline", [])
    expanded_list = runs.get("expanded", [])
    baseline = baseline_list[0] if baseline_list else None
    expanded = expanded_list[0] if expanded_list else None
    return baseline, expanded


def load_and_merge_predictions(
    results_dir: str,
    baseline_run: str,
    expanded_run: str,
    model: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load preds_pooled from both runs, merge on (Year, WEOCountryCode), return
    (y_true, p_baseline, p_expanded) or None if merge yields no common rows.
    """
    base_path = os.path.join(
        results_dir, baseline_run, "predictions", model, "preds_pooled.csv"
    )
    exp_path = os.path.join(
        results_dir, expanded_run, "predictions", model, "preds_pooled.csv"
    )
    df_b = pd.read_csv(base_path)
    df_e = pd.read_csv(exp_path)
    df_b = df_b.rename(columns={"y_pred": "y_pred_baseline"})
    df_e = df_e.rename(columns={"y_pred": "y_pred_expanded"})
    merge_cols = ["Year", "WEOCountryCode"]
    df_b = df_b[merge_cols + ["y_pred_baseline"]]
    df_e = df_e[merge_cols + ["y_true", "y_pred_expanded"]]
    merged = pd.merge(df_b, df_e, on=merge_cols, how="inner")
    merged = merged.dropna(subset=["y_true", "y_pred_baseline", "y_pred_expanded"])
    if len(merged) == 0:
        return None
    y_true = merged["y_true"].values
    p_baseline = merged["y_pred_baseline"].values
    p_expanded = merged["y_pred_expanded"].values
    return y_true, p_baseline, p_expanded


def build_comparison_table(
    results_dir: str,
    baseline_run: str,
    expanded_run: str,
    models: List[str],
) -> pd.DataFrame:
    """One table per (horizon, group): rows=models, columns AUC/F1 and winners."""
    rows = []
    for model in models:
        aligned = load_and_merge_predictions(results_dir, baseline_run, expanded_run, model)
        if aligned is None:
            rows.append(
                {
                    "Model": model,
                    "AUC_baseline": np.nan,
                    "AUC_expanded": np.nan,
                    "AUC_diff": np.nan,
                    "AUC_winner": "—",
                    "DeLong_p": np.nan,
                    "AUC_sig": False,
                    "F1_baseline": np.nan,
                    "F1_expanded": np.nan,
                    "F1_diff": np.nan,
                    "F1_winner": "—",
                    "Bootstrap_p": np.nan,
                    "F1_sig": False,
                }
            )
            continue
        y_true, p_baseline, p_expanded = aligned
        # Positive diff = expanded better (p1=expanded, p2=baseline)
        d_auc = delong_test(y_true, p_expanded, p_baseline, alpha=ALPHA)
        d_f1 = bootstrap_f1_test(
            y_true, p_expanded, p_baseline,
            n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE, alpha=ALPHA,
        )
        auc_b, auc_e = d_auc["auc_2"], d_auc["auc_1"]
        f1_b, f1_e = d_f1["f1_2"], d_f1["f1_1"]
        auc_diff = d_auc["diff"]
        f1_diff = d_f1["diff"]
        # Winner: Baseline / Expanded / Tie; append * when significant
        if d_auc["significant"]:
            auc_winner = "Expanded*" if auc_diff > 0 else "Baseline*"
        else:
            auc_winner = "Expanded" if auc_diff > 0 else ("Baseline" if auc_diff < 0 else "Tie")
        if d_f1["significant"]:
            f1_winner = "Expanded*" if f1_diff > 0 else "Baseline*"
        else:
            f1_winner = "Expanded" if f1_diff > 0 else ("Baseline" if f1_diff < 0 else "Tie")
        rows.append(
            {
                "Model": model,
                "AUC_baseline": auc_b,
                "AUC_expanded": auc_e,
                "AUC_diff": auc_diff,
                "AUC_winner": auc_winner,
                "DeLong_p": d_auc["p_value"],
                "AUC_sig": d_auc["significant"],
                "F1_baseline": f1_b,
                "F1_expanded": f1_e,
                "F1_diff": f1_diff,
                "F1_winner": f1_winner,
                "Bootstrap_p": d_f1["p_value"],
                "F1_sig": d_f1["significant"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    results_dir = RESULTS_DIR
    output_dir = os.path.join(results_dir, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    runs = discover_runs(results_dir)
    if not runs:
        print("No valid runs found under", results_dir)
        return

    # Infer models from first available run that has predictions
    models = DEFAULT_MODELS

    summary_lines = []
    for (horizon, group), run_lists in sorted(runs.items()):
        baseline_run, expanded_run = pick_per_dataset(run_lists)
        if baseline_run is None:
            print(f"Warning: No baseline run for h={horizon} {group}; skipping.")
            continue
        if expanded_run is None:
            print(f"Warning: No expanded run for h={horizon} {group}; skipping.")
            continue
        print(f"Comparing h={horizon} {group}: baseline={baseline_run}, expanded={expanded_run}")
        table = build_comparison_table(results_dir, baseline_run, expanded_run, models)
        out_name = f"comparison_h{horizon}_{group}.csv"
        out_path = os.path.join(output_dir, out_name)
        table.to_csv(out_path, index=False)
        print(f"  -> {out_path}")
        summary_lines.append(f"## h={horizon} {group}\n")
        summary_lines.append(table.to_string(index=False) + "\n\n")

    # Optional: one combined summary markdown
    summary_path = os.path.join(output_dir, "comparison_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Baseline vs Expanded comparison\n\n")
        f.write("\n".join(summary_lines))
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
