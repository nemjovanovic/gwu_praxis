# ================================================================
# data_transform.py
# ================================================================
#
# Reads existing train/test CSVs from data/final, trains LSTM and
# PatchTST forecasters for each year, generates forecast features,
# and writes augmented train/test CSVs to traintest_transforms/.
#
# CLI:
#   python -m superlearners.data_transform \
#       --dataset baseline --horizon 2 --group ALL
# ================================================================

import os
import sys
import argparse
import functools
import time
import warnings
from datetime import datetime

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from superlearners.config import (
    SPLITS_BASE,
    TRANSFORMS_DIR,
    VALID_DATASETS,
    VALID_HORIZONS,
    VALID_GROUPS,
    FORECAST_VARS,
    SEQUENCE_LEN,
    NON_FEATURE_COLS,
    TARGET_COL,
    COUNTRY_COL,
    YEAR_COL,
    T_MIN,
    get_t_max,
    get_forecast_feature_names,
)

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------
# helpers: discover years, load split (same logic as basemodels)
# ----------------------------------------------------------------

def get_available_years(dataset: str, horizon: int, group: str):
    """Return sorted list of years that have both train and test files."""
    train_dir = os.path.join(SPLITS_BASE, "train")
    test_dir = os.path.join(SPLITS_BASE, "test")
    t_max = get_t_max(horizon)
    years = []
    for t in range(T_MIN, t_max + 1):
        tp = os.path.join(train_dir, f"train_{dataset}_h{horizon}_{group}_t{t}.csv")
        ep = os.path.join(test_dir, f"test_{dataset}_h{horizon}_{group}_t{t}.csv")
        if os.path.exists(tp) and os.path.exists(ep):
            years.append(t)
    return sorted(years)


def load_split(dataset: str, horizon: int, group: str, year: int):
    train_path = os.path.join(SPLITS_BASE, "train", f"train_{dataset}_h{horizon}_{group}_t{year}.csv")
    test_path = os.path.join(SPLITS_BASE, "test", f"test_{dataset}_h{horizon}_{group}_t{year}.csv")
    return pd.read_csv(train_path), pd.read_csv(test_path)


# ----------------------------------------------------------------
# sequence building (from train_df only)
# ----------------------------------------------------------------

def build_sequences(df: pd.DataFrame, seq_len: int, forecast_vars):
    """
    Build sequences of length seq_len per country from a DataFrame.
    Returns X_seq (n, seq_len, n_vars), y_next (n, n_vars).
    """
    sequences, targets = [], []
    for country in df[COUNTRY_COL].unique():
        cdf = df[df[COUNTRY_COL] == country].sort_values(YEAR_COL)
        if len(cdf) < seq_len + 1:
            continue
        vals = cdf[forecast_vars].values.astype(np.float32)
        for i in range(len(vals) - seq_len):
            sequences.append(vals[i:i + seq_len])
            targets.append(vals[i + seq_len])
    if not sequences:
        nv = len(forecast_vars)
        return np.empty((0, seq_len, nv), dtype=np.float32), np.empty((0, nv), dtype=np.float32)
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


# ----------------------------------------------------------------
# generate forecast features for a DataFrame
# ----------------------------------------------------------------

def generate_forecast_features_for_df(
    forecaster, df: pd.DataFrame, train_df: pd.DataFrame,
    seq_len: int, forecast_vars,
):
    """
    For every row in *df*, build the most recent seq_len-length sequence
    from *train_df* history for that country, run the forecaster, and
    produce forecast + change columns.  Rows that lack enough history
    get zeros.

    Returns a DataFrame with 2*len(forecast_vars) new columns (same index as df).
    """
    feature_names = forecaster.get_feature_names()
    result = pd.DataFrame(0.0, index=df.index, columns=feature_names)

    # pre-build per-country history from train_df
    history_cache = {}
    for country, grp in train_df.groupby(COUNTRY_COL):
        history_cache[country] = grp.sort_values(YEAR_COL)[forecast_vars].values.astype(np.float32)

    for idx, row in df.iterrows():
        country = row[COUNTRY_COL]
        hist = history_cache.get(country)
        if hist is None or len(hist) < seq_len:
            continue
        # use last seq_len rows of the country's training history
        seq = hist[-seq_len:].reshape(1, seq_len, -1)
        current = hist[-1].reshape(1, -1)
        try:
            feats = forecaster.forecast_features(current, seq)  # (1, 2*n_vars)
            for j, col in enumerate(feature_names):
                result.at[idx, col] = feats[0, j]
        except Exception as e:
            print(f"    WARNING: forecast failed for country={country}, idx={idx}: {e}")

    return result


# ----------------------------------------------------------------
# main transform logic
# ----------------------------------------------------------------

def transform(dataset: str, horizon: int, group: str):
    years = get_available_years(dataset, horizon, group)
    if not years:
        print(f"No splits found for {dataset} h{horizon} {group}")
        return

    out_train_dir = os.path.join(TRANSFORMS_DIR, "train")
    out_test_dir = os.path.join(TRANSFORMS_DIR, "test")
    out_manifest_dir = os.path.join(TRANSFORMS_DIR, "manifest")
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)
    os.makedirs(out_manifest_dir, exist_ok=True)

    # lazy imports so TF is only loaded when this module actually runs
    from superlearners.models.lstm_model import LSTMForecaster
    from superlearners.models.patchtst_model import PatchTSTForecaster

    manifest_rows = []

    for year in years:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  Year {year}  ({dataset} h{horizon} {group})")
        print(f"{'='*60}")

        train_df, test_df = load_split(dataset, horizon, group, year)
        print(f"  train rows={len(train_df)}, test rows={len(test_df)}")

        # --- build sequences from train (already winsorized in make_splits) ---
        X_seq, y_next = build_sequences(train_df, SEQUENCE_LEN, FORECAST_VARS)
        print(f"  sequences built: {X_seq.shape[0]}")

        if X_seq.shape[0] == 0:
            print(f"  ERROR: 0 sequences built! No country has > {SEQUENCE_LEN} rows.")
            print(f"         Max rows/country in train: {train_df.groupby(COUNTRY_COL).size().max()}")
            print(f"         Need at least {SEQUENCE_LEN + 1} rows to build 1 sequence.")
            print(f"         Consider reducing SEQUENCE_LEN (currently {SEQUENCE_LEN}).")
            raise ValueError(
                f"Cannot train forecasters: 0 sequences built for year {year}. "
                f"SEQUENCE_LEN={SEQUENCE_LEN} is too large for the data."
            )
        if X_seq.shape[0] < 20:
            print(f"  WARNING: only {X_seq.shape[0]} sequences; forecast features may be noisy")

        # 90/10 split for validation
        n_val = max(1, int(0.1 * len(X_seq)))
        X_val, y_val = X_seq[-n_val:], y_next[-n_val:]
        X_tr, y_tr = X_seq[:-n_val], y_next[:-n_val]

        # --- train LSTM ---
        print("  Training LSTM forecaster...")
        lstm = LSTMForecaster()
        lstm.fit(X_tr, y_tr, X_val, y_val, verbose=True)

        # --- train PatchTST ---
        print("  Training PatchTST forecaster...")
        patchtst = PatchTSTForecaster()
        patchtst.fit(X_tr, y_tr, X_val, y_val, verbose=True)

        # --- generate forecast features (data already winsorized by make_splits) ---
        print("  Generating forecast features for train...")
        lstm_train_feats = generate_forecast_features_for_df(
            lstm, train_df, train_df, SEQUENCE_LEN, FORECAST_VARS,
        )
        patchtst_train_feats = generate_forecast_features_for_df(
            patchtst, train_df, train_df, SEQUENCE_LEN, FORECAST_VARS,
        )

        print("  Generating forecast features for test...")
        lstm_test_feats = generate_forecast_features_for_df(
            lstm, test_df, train_df, SEQUENCE_LEN, FORECAST_VARS,
        )
        patchtst_test_feats = generate_forecast_features_for_df(
            patchtst, test_df, train_df, SEQUENCE_LEN, FORECAST_VARS,
        )

        # --- augment and save ---
        aug_train = pd.concat([train_df, lstm_train_feats, patchtst_train_feats], axis=1)
        aug_test = pd.concat([test_df, lstm_test_feats, patchtst_test_feats], axis=1)

        train_out = os.path.join(out_train_dir, f"train_{dataset}_h{horizon}_{group}_t{year}.csv")
        test_out = os.path.join(out_test_dir, f"test_{dataset}_h{horizon}_{group}_t{year}.csv")
        aug_train.to_csv(train_out, index=False)
        aug_test.to_csv(test_out, index=False)
        elapsed = time.time() - t0
        print(f"  Saved augmented CSVs ({elapsed:.1f}s)")

        # manifest row
        train_crisis = aug_train[TARGET_COL].mean() if len(aug_train) else 0
        test_crisis = aug_test[TARGET_COL].mean() if len(aug_test) else 0
        manifest_rows.append({
            "Dataset": dataset,
            "Horizon": horizon,
            "Group": group,
            "t": year,
            "TrainRows": len(aug_train),
            "TestRows": len(aug_test),
            "TrainCrisisRate": train_crisis,
            "TestCrisisRate": test_crisis,
            "TrainPath": train_out,
            "TestPath": test_out,
        })

    # write manifest
    manifest_path = os.path.join(out_manifest_dir, f"manifest_{dataset}_h{horizon}_{group}.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nManifest: {manifest_path}")
    print("Transform complete.")


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform data/final splits into forecast-augmented CSVs for superlearners"
    )
    parser.add_argument("--dataset", type=str, choices=VALID_DATASETS, required=True)
    parser.add_argument("--horizon", type=int, choices=VALID_HORIZONS, required=True)
    parser.add_argument("--group", type=str, choices=VALID_GROUPS, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Superlearners data_transform")
    print(f"  dataset={args.dataset}  horizon={args.horizon}  group={args.group}")
    print("=" * 60)
    transform(args.dataset, args.horizon, args.group)


if __name__ == "__main__":
    main()
