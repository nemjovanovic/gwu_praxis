# ================================================================
# evaluate_basemodel_supermodel.py
# ================================================================
#
# Compare superlearner models (RF_LSTM, RF_PatchTST) against
# basemodels (RuleOfThumb, Probit, RandomForest, AdaBoost, XGBoost)
# on the same (dataset, horizon, group) using DeLong (AUC) and
# paired bootstrap (F1) significance tests.
#
# CLI:
#   python -m superlearners.evaluate_basemodel_supermodel \
#       --dataset baseline --horizon 2 --group ALL
# ================================================================

import os
import sys
import argparse
import warnings
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from superlearners.config import (
    RESULTS_DIR as SL_RESULTS_DIR,
    RESULTS_COMPARE_DIR,
    VALID_DATASETS,
    VALID_HORIZONS,
    VALID_GROUPS,
    SL_MODEL_NAMES,
)
from basemodels.config import RESULTS_DIR as BM_RESULTS_DIR
from metrics.significance_tests import delong_test, bootstrap_f1_test

warnings.filterwarnings("ignore")

BASEMODEL_NAMES = ["RuleOfThumb", "Probit", "RandomForest", "AdaBoost", "XGBoost"]
ALPHA = 0.05
N_BOOTSTRAP = 1000
RANDOM_STATE = 1234

# ----------------------------------------------------------------
# Run discovery
# ----------------------------------------------------------------

def _find_run(results_dir: str, run_name: str, model_names: List[str]):
    """Find run directory for (dataset, horizon, group) that has preds_pooled for all models."""
    run_path = os.path.join(results_dir, run_name)
    if not os.path.isdir(run_path):
        return None
    for mn in model_names:
        pp = os.path.join(run_path, "predictions", mn, "preds_pooled.csv")
        if not os.path.isfile(pp):
            return None
    return run_name


def _load_pooled(results_dir: str, run_name: str, model: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(results_dir, run_name, "predictions", model, "preds_pooled.csv"))


# ----------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------

def compare(dataset: str, horizon: int, group: str):
    sl_run_name = f"superlearner_{dataset}_h{horizon}_{group}"
    bm_run_name = f"{dataset}_h{horizon}_{group}"
    sl_run = _find_run(SL_RESULTS_DIR, sl_run_name, SL_MODEL_NAMES)
    bm_run = _find_run(BM_RESULTS_DIR, bm_run_name, BASEMODEL_NAMES)

    if sl_run is None:
        print(f"No superlearner run found for {dataset} h{horizon} {group}. Run superlearners first.")
        return
    if bm_run is None:
        print(f"No basemodel run found for {dataset} h{horizon} {group}. Run basemodels first.")
        return

    print(f"Superlearner run: {sl_run}")
    print(f"Basemodel   run: {bm_run}")

    os.makedirs(RESULTS_COMPARE_DIR, exist_ok=True)

    rows = []
    for sl_model in SL_MODEL_NAMES:
        sl_df = _load_pooled(SL_RESULTS_DIR, sl_run, sl_model)
        sl_df = sl_df.rename(columns={"y_pred": "y_pred_super"})

        for bm_model in BASEMODEL_NAMES:
            bm_df = _load_pooled(BM_RESULTS_DIR, bm_run, bm_model)
            bm_df = bm_df.rename(columns={"y_pred": "y_pred_base"})

            merge_cols = ["Year", "WEOCountryCode"]
            merged = pd.merge(
                sl_df[merge_cols + ["y_true", "y_pred_super"]],
                bm_df[merge_cols + ["y_pred_base"]],
                on=merge_cols, how="inner",
            )
            merged = merged.dropna(subset=["y_true", "y_pred_super", "y_pred_base"])

            if len(merged) == 0:
                rows.append({
                    "Model_Super": sl_model, "Model_Base": bm_model,
                    "AUC_base": np.nan, "AUC_super": np.nan, "AUC_diff": np.nan,
                    "AUC_winner": "—", "DeLong_p": np.nan,
                    "F1_base": np.nan, "F1_super": np.nan, "F1_diff": np.nan,
                    "F1_winner": "—", "Bootstrap_p": np.nan,
                })
                continue

            y_true = merged["y_true"].values
            p_super = merged["y_pred_super"].values
            p_base = merged["y_pred_base"].values

            # positive diff = superlearner better
            d_auc = delong_test(y_true, p_super, p_base, alpha=ALPHA)
            d_f1 = bootstrap_f1_test(y_true, p_super, p_base,
                                      n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE, alpha=ALPHA)

            auc_b, auc_s = d_auc["auc_2"], d_auc["auc_1"]
            f1_b, f1_s = d_f1["f1_2"], d_f1["f1_1"]
            auc_diff = d_auc["diff"]
            f1_diff = d_f1["diff"]

            if d_auc["significant"]:
                auc_winner = "Superlearner*" if auc_diff > 0 else "Basemodel*"
            else:
                auc_winner = "Superlearner" if auc_diff > 0 else ("Basemodel" if auc_diff < 0 else "Tie")

            if d_f1["significant"]:
                f1_winner = "Superlearner*" if f1_diff > 0 else "Basemodel*"
            else:
                f1_winner = "Superlearner" if f1_diff > 0 else ("Basemodel" if f1_diff < 0 else "Tie")

            rows.append({
                "Model_Super": sl_model, "Model_Base": bm_model,
                "AUC_base": auc_b, "AUC_super": auc_s, "AUC_diff": auc_diff,
                "AUC_winner": auc_winner, "DeLong_p": d_auc["p_value"],
                "F1_base": f1_b, "F1_super": f1_s, "F1_diff": f1_diff,
                "F1_winner": f1_winner, "Bootstrap_p": d_f1["p_value"],
            })

    table = pd.DataFrame(rows)

    csv_path = os.path.join(RESULTS_COMPARE_DIR, f"comparison_{dataset}_h{horizon}_{group}.csv")
    table.to_csv(csv_path, index=False)
    print(f"\nCSV: {csv_path}")

    # markdown summary
    md_lines = [f"# Superlearner vs Basemodel comparison\n"]
    md_lines.append(f"**Dataset:** {dataset}  **Horizon:** h={horizon}  **Group:** {group}\n")
    md_lines.append(f"**Superlearner run:** {sl_run}\n")
    md_lines.append(f"**Basemodel run:** {bm_run}\n\n")
    md_lines.append(table.to_string(index=False))
    md_lines.append("")

    md_path = os.path.join(RESULTS_COMPARE_DIR, f"comparison_summary_{dataset}_h{horizon}_{group}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Summary: {md_path}")


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compare superlearner vs basemodel results with significance tests")
    p.add_argument("--dataset", type=str, choices=VALID_DATASETS, required=True)
    p.add_argument("--horizon", type=int, choices=VALID_HORIZONS, required=True)
    p.add_argument("--group", type=str, choices=VALID_GROUPS, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Superlearner vs Basemodel Evaluation")
    print(f"  dataset={args.dataset}  horizon={args.horizon}  group={args.group}")
    print("=" * 60)
    compare(args.dataset, args.horizon, args.group)


if __name__ == "__main__":
    main()
