# ================================================================
# evaluate_stacked_vs_basemodel.py (Stacked Super Learner)
# ================================================================
#
# Compare stacked super learner models (SL_Logistic and its 4
# level-0 models) against the 5 basemodels using DeLong (AUC)
# and paired bootstrap (F1) significance tests.
#
# CLI:
#   python -m superlearner_stacked.evaluate_stacked_vs_basemodel \
#       --dataset baseline --horizon 2 --group ALL
#
# ================================================================

import os
import sys
import argparse
import warnings
from typing import List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from basemodels.config import (
    VALID_DATASETS,
    VALID_HORIZONS,
    VALID_GROUPS,
    RESULTS_DIR as BM_RESULTS_DIR,
)
from superlearner_stacked.config import (
    RESULTS_DIR as STACKED_RESULTS_DIR,
    RESULTS_COMPARE_DIR,
    LEVEL0_MODEL_NAMES,
    META_LEARNER_NAMES,
)
from metrics.significance_tests import delong_test, bootstrap_f1_test

warnings.filterwarnings("ignore")

# All stacked models to compare (meta-learner first, then level-0)
STACKED_ALL_MODELS = META_LEARNER_NAMES + LEVEL0_MODEL_NAMES
BASEMODEL_NAMES = ["RuleOfThumb", "Probit", "RandomForest", "AdaBoost", "XGBoost"]
ALPHA = 0.05
N_BOOTSTRAP = 1000
RANDOM_STATE = 1234


# ----------------------------------------------------------------
# Run discovery
# ----------------------------------------------------------------

def _find_run(
    results_dir: str, run_name: str, model_names: List[str],
) -> Optional[str]:
    """Find run directory that has preds_pooled for all models."""
    run_path = os.path.join(results_dir, run_name)
    if not os.path.isdir(run_path):
        return None
    for mn in model_names:
        pp = os.path.join(run_path, "predictions", mn, "preds_pooled.csv")
        if not os.path.isfile(pp):
            return None
    return run_name


def _load_pooled(results_dir: str, run_name: str, model: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(results_dir, run_name, "predictions", model, "preds_pooled.csv")
    )


# ----------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------

def compare(dataset: str, horizon: int, group: str):
    stacked_run_name = f"superlearner_stacked_{dataset}_h{horizon}_{group}"
    bm_run_name = f"{dataset}_h{horizon}_{group}"

    stacked_run = _find_run(STACKED_RESULTS_DIR, stacked_run_name, STACKED_ALL_MODELS)
    bm_run = _find_run(BM_RESULTS_DIR, bm_run_name, BASEMODEL_NAMES)

    if stacked_run is None:
        print(f"No stacked super learner run found for {dataset} h{horizon} {group}.")
        print(f"Expected: {os.path.join(STACKED_RESULTS_DIR, stacked_run_name)}")
        print("Run superlearner_stacked/run_stacked.py first.")
        return
    if bm_run is None:
        print(f"No basemodel run found for {dataset} h{horizon} {group}.")
        print(f"Expected: {os.path.join(BM_RESULTS_DIR, bm_run_name)}")
        print("Run basemodels/run_basemodels.py first.")
        return

    print(f"Stacked run:  {stacked_run}")
    print(f"Basemodel run: {bm_run}")

    os.makedirs(RESULTS_COMPARE_DIR, exist_ok=True)

    rows = []
    for stacked_model in STACKED_ALL_MODELS:
        sl_df = _load_pooled(STACKED_RESULTS_DIR, stacked_run, stacked_model)
        sl_df = sl_df.rename(columns={"y_pred": "y_pred_stacked"})

        for bm_model in BASEMODEL_NAMES:
            bm_df = _load_pooled(BM_RESULTS_DIR, bm_run, bm_model)
            bm_df = bm_df.rename(columns={"y_pred": "y_pred_base"})

            merge_cols = ["Year", "WEOCountryCode"]
            merged = pd.merge(
                sl_df[merge_cols + ["y_true", "y_pred_stacked"]],
                bm_df[merge_cols + ["y_pred_base"]],
                on=merge_cols, how="inner",
            )
            merged = merged.dropna(
                subset=["y_true", "y_pred_stacked", "y_pred_base"]
            )

            if len(merged) == 0:
                rows.append({
                    "Model_Stacked": stacked_model, "Model_Base": bm_model,
                    "AUC_base": np.nan, "AUC_stacked": np.nan, "AUC_diff": np.nan,
                    "AUC_winner": "—", "DeLong_p": np.nan,
                    "F1_base": np.nan, "F1_stacked": np.nan, "F1_diff": np.nan,
                    "F1_winner": "—", "Bootstrap_p": np.nan,
                })
                continue

            y_true = merged["y_true"].values
            p_stacked = merged["y_pred_stacked"].values
            p_base = merged["y_pred_base"].values

            # Positive diff = stacked better (p1=stacked, p2=base)
            d_auc = delong_test(y_true, p_stacked, p_base, alpha=ALPHA)
            d_f1 = bootstrap_f1_test(
                y_true, p_stacked, p_base,
                n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE,
                alpha=ALPHA,
            )

            auc_b, auc_s = d_auc["auc_2"], d_auc["auc_1"]
            f1_b, f1_s = d_f1["f1_2"], d_f1["f1_1"]
            auc_diff = d_auc["diff"]
            f1_diff = d_f1["diff"]

            if d_auc["significant"]:
                auc_winner = "Stacked*" if auc_diff > 0 else "Basemodel*"
            else:
                auc_winner = (
                    "Stacked" if auc_diff > 0
                    else ("Basemodel" if auc_diff < 0 else "Tie")
                )

            if d_f1["significant"]:
                f1_winner = "Stacked*" if f1_diff > 0 else "Basemodel*"
            else:
                f1_winner = (
                    "Stacked" if f1_diff > 0
                    else ("Basemodel" if f1_diff < 0 else "Tie")
                )

            rows.append({
                "Model_Stacked": stacked_model, "Model_Base": bm_model,
                "AUC_base": auc_b, "AUC_stacked": auc_s, "AUC_diff": auc_diff,
                "AUC_winner": auc_winner, "DeLong_p": d_auc["p_value"],
                "F1_base": f1_b, "F1_stacked": f1_s, "F1_diff": f1_diff,
                "F1_winner": f1_winner, "Bootstrap_p": d_f1["p_value"],
            })

    table = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(
        RESULTS_COMPARE_DIR,
        f"comparison_{dataset}_h{horizon}_{group}.csv",
    )
    table.to_csv(csv_path, index=False)
    print(f"\nCSV: {csv_path}")

    # Markdown summary
    md_lines = ["# Stacked Super Learner vs Basemodel Comparison\n"]
    md_lines.append(
        f"**Dataset:** {dataset}  **Horizon:** h={horizon}  **Group:** {group}\n"
    )
    md_lines.append(f"**Stacked run:** {stacked_run}\n")
    md_lines.append(f"**Basemodel run:** {bm_run}\n\n")
    md_lines.append(table.to_string(index=False))
    md_lines.append("")

    md_path = os.path.join(
        RESULTS_COMPARE_DIR,
        f"comparison_summary_{dataset}_h{horizon}_{group}.md",
    )
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Summary: {md_path}")


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare stacked super learner vs basemodel results"
    )
    p.add_argument(
        "--dataset", type=str, choices=VALID_DATASETS, required=True,
    )
    p.add_argument(
        "--horizon", type=int, choices=VALID_HORIZONS, required=True,
    )
    p.add_argument(
        "--group", type=str, choices=VALID_GROUPS, required=True,
    )
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Stacked Super Learner vs Basemodel Evaluation")
    print(f"  dataset={args.dataset}  horizon={args.horizon}  group={args.group}")
    print("=" * 60)
    compare(args.dataset, args.horizon, args.group)


if __name__ == "__main__":
    main()
