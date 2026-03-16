# ================================================================
# evaluate_shap_vs_basemodel.py (SHAP Model)
# ================================================================
#
# Compare SHAP-selected RF models (RF_SHAP_5, RF_SHAP_10) against
# basemodels (RuleOfThumb, Probit, RandomForest, AdaBoost, XGBoost)
# on the same (horizon, group) using DeLong (AUC) and paired
# bootstrap (F1) significance tests.
#
# CLI:
#   python -m shap_model.evaluate_shap_vs_basemodel \
#       --dataset baseline --horizon 2 --group ALL
#
#   python -m shap_model.evaluate_shap_vs_basemodel \
#       --dataset expanded --horizon 2 --group ALL
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

from shap_model.config import (
    RESULTS_DIR as SHAP_RESULTS_DIR,
    RESULTS_COMPARE_DIR,
    VALID_DATASETS,
    VALID_HORIZONS,
    VALID_GROUPS,
    MODEL_NAMES as SHAP_MODEL_NAMES,
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

def _find_run(
    results_dir: str, run_name: str, model_names: List[str]
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
    shap_run_name = f"shap_h{horizon}_{group}"
    bm_run_name = f"{dataset}_h{horizon}_{group}"

    shap_run = _find_run(SHAP_RESULTS_DIR, shap_run_name, SHAP_MODEL_NAMES)
    bm_run = _find_run(BM_RESULTS_DIR, bm_run_name, BASEMODEL_NAMES)

    if shap_run is None:
        print(
            f"No SHAP model run found for h{horizon} {group}. "
            f"Run shap_model first: python -m shap_model.run_shap_model --horizon {horizon} --group {group}"
        )
        return
    if bm_run is None:
        print(
            f"No basemodel run found for {dataset} h{horizon} {group}. "
            f"Run basemodels first."
        )
        return

    print(f"SHAP model run: {shap_run}")
    print(f"Basemodel  run: {bm_run}")

    os.makedirs(RESULTS_COMPARE_DIR, exist_ok=True)

    rows = []
    for shap_model in SHAP_MODEL_NAMES:
        shap_df = _load_pooled(SHAP_RESULTS_DIR, shap_run, shap_model)
        shap_df = shap_df.rename(columns={"y_pred": "y_pred_shap"})

        for bm_model in BASEMODEL_NAMES:
            bm_df = _load_pooled(BM_RESULTS_DIR, bm_run, bm_model)
            bm_df = bm_df.rename(columns={"y_pred": "y_pred_base"})

            merge_cols = ["Year", "WEOCountryCode"]
            merged = pd.merge(
                shap_df[merge_cols + ["y_true", "y_pred_shap"]],
                bm_df[merge_cols + ["y_pred_base"]],
                on=merge_cols,
                how="inner",
            )
            merged = merged.dropna(
                subset=["y_true", "y_pred_shap", "y_pred_base"]
            )

            if len(merged) == 0:
                rows.append({
                    "Model_SHAP": shap_model,
                    "Model_Base": bm_model,
                    "AUC_base": np.nan,
                    "AUC_shap": np.nan,
                    "AUC_diff": np.nan,
                    "AUC_winner": "—",
                    "DeLong_p": np.nan,
                    "F1_base": np.nan,
                    "F1_shap": np.nan,
                    "F1_diff": np.nan,
                    "F1_winner": "—",
                    "Bootstrap_p": np.nan,
                })
                continue

            y_true = merged["y_true"].values
            p_shap = merged["y_pred_shap"].values
            p_base = merged["y_pred_base"].values

            d_auc = delong_test(y_true, p_shap, p_base, alpha=ALPHA)
            d_f1 = bootstrap_f1_test(
                y_true, p_shap, p_base,
                n_bootstrap=N_BOOTSTRAP,
                random_state=RANDOM_STATE,
                alpha=ALPHA,
            )

            auc_b, auc_s = d_auc["auc_2"], d_auc["auc_1"]
            f1_b, f1_s = d_f1["f1_2"], d_f1["f1_1"]
            auc_diff = d_auc["diff"]
            f1_diff = d_f1["diff"]

            if d_auc["significant"]:
                auc_winner = "SHAP_Model*" if auc_diff > 0 else "Basemodel*"
            else:
                auc_winner = (
                    "SHAP_Model" if auc_diff > 0
                    else ("Basemodel" if auc_diff < 0 else "Tie")
                )

            if d_f1["significant"]:
                f1_winner = "SHAP_Model*" if f1_diff > 0 else "Basemodel*"
            else:
                f1_winner = (
                    "SHAP_Model" if f1_diff > 0
                    else ("Basemodel" if f1_diff < 0 else "Tie")
                )

            rows.append({
                "Model_SHAP": shap_model,
                "Model_Base": bm_model,
                "AUC_base": auc_b,
                "AUC_shap": auc_s,
                "AUC_diff": auc_diff,
                "AUC_winner": auc_winner,
                "DeLong_p": d_auc["p_value"],
                "F1_base": f1_b,
                "F1_shap": f1_s,
                "F1_diff": f1_diff,
                "F1_winner": f1_winner,
                "Bootstrap_p": d_f1["p_value"],
            })

    table = pd.DataFrame(rows)

    csv_path = os.path.join(
        RESULTS_COMPARE_DIR,
        f"comparison_{dataset}_h{horizon}_{group}.csv",
    )
    table.to_csv(csv_path, index=False)
    print(f"\nCSV: {csv_path}")

    # Markdown summary
    md_lines = [
        f"# SHAP Model vs Basemodel Comparison\n",
        f"**Dataset:** {dataset}  **Horizon:** h={horizon}  **Group:** {group}\n",
        f"**SHAP model run:** {shap_run}\n",
        f"**Basemodel run:** {bm_run}\n\n",
        table.to_string(index=False),
        "",
    ]

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
        description="Compare SHAP-selected RF models vs basemodels"
    )
    p.add_argument(
        "--dataset", type=str, choices=VALID_DATASETS, required=True,
        help="Basemodel dataset to compare against: baseline or expanded",
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
    print("SHAP Model vs Basemodel Evaluation")
    print(f"  dataset={args.dataset}  horizon={args.horizon}  group={args.group}")
    print("=" * 60)
    compare(args.dataset, args.horizon, args.group)


if __name__ == "__main__":
    main()
