# =====================================================
# make_splits.py
#
# Reads unimputed constructed datasets, produces horizon-specific
# datasets and rolling-window train/test CSVs under data/final/.
#
# Per-split preprocessing (leak-free):
#   1. Winsorize: bounds computed from train only, applied to train+test
#   2. Impute:    fit on train only, applied to train+test
#
# Terminology:
#   t  = feature/observation year (the row year).
#        crisis label at year t is forward-looking: crisis=1 if any
#        crisis in [t+1, t+h].
#   Train window for split year t: years [t - h - LOOKBACK + 1, t - h]
#   Test: year t
#
# Inputs: data/constructed_datasets/
#   - baseline_without_imputation.csv
#   - expanded_without_imputation.csv
#
# Outputs: data/final/
#   - horizonsplit/{dataset}_h{h}_{group}.csv  (raw, unwinsorized, unimputed)
#   - train/train_{dataset}_h{h}_{group}_t{t}.csv
#   - test/test_{dataset}_h{h}_{group}_t{t}.csv
#   - manifest/manifest_{dataset}_h{h}_{group}.csv
# =====================================================

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ----------------------------
# Configuration
# ----------------------------
DATA_DIR = Path(__file__).resolve().parent
CONSTRUCTED_DIR = DATA_DIR / "constructed_datasets"
OUT_BASE = DATA_DIR / "final"

DATASET_KEYS = ["baseline", "expanded"]

# Map short key -> constructed_datasets filename
DATASET_FILES: Dict[str, str] = {
    "baseline": "baseline_without_imputation.csv",
    "expanded": "expanded_without_imputation.csv",
}

HORIZONS = [2, 5, 10]
GROUPS = ["ALL", "EME", "LIC"]
LOOKBACK = 10
T_MIN = 2000


def t_max_fn(h: int) -> int:
    return 2021 - h


KEY_COLS = ["WEOCountryCode", "GroupName", "Year", "crisis"]

# ----------------------------
# Imputation column groups
# ----------------------------
# Baseline columns
BASELINE_GROUPNAME_ONLY = ["D", "ED", "DSED", "PB", "r_g", "Rem"]
BASELINE_GROUPNAME_YEAR = ["CPI", "DlnREER", "CA", "ka_open", "gee", "FXR", "lnGDPpc"]
BASELINE_GLOBAL_MEDIAN = ["lnVIX", "WGDPg", "Rshort", "Rlong", "DlnPoil", "DlnPcom"]

# Expanded-only columns
EXPANDED_GROUPNAME_ONLY = [
    "dep_old", "dep_totl", "fh_pr", "van_index", "cbie_lvau",
    "wdi_birthskill", "free_ovr", "free_civ", "crd_priv", "M2_GDP", "govtax_GDP",
]
EXPANDED_GROUPNAME_YEAR = [
    "deflator", "cons_GDP", "finv_GDP", "exports_GDP", "imports_GDP",
    "cbrate", "strate", "unemp", "clm_govt", "clm_priv", "net_oda",
]

# Columns to exclude from winsorization
WINSORIZE_EXCLUDE = {"WEOCountryCode", "GroupName", "Year", "crisis", "c", "CH", "TC"}

# Winsorization percentile (1 = 1st/99th)
WINSORIZE_TRIM = 1


# ----------------------------
# Helpers: forward crisis label and horizon preparation
# ----------------------------
def add_forward_crisis_labels(
    df: pd.DataFrame,
    h: int,
    crisis_col: str = "crisis",
    country_col: str = "WEOCountryCode",
) -> pd.DataFrame:
    """
    For each country-year:
      - c = current-year crisis (copy of original crisis_col)
      - crisis = 1 if any crisis occurs in the next h years (t+1 ... t+h), else 0
    """
    df = df.copy()
    df[country_col] = df[country_col].astype(str)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.sort_values([country_col, "Year"]).reset_index(drop=True)

    df["c"] = df[crisis_col].fillna(0).astype(int)
    df["crisis"] = 0

    for cid, idx in df.groupby(country_col).groups.items():
        g = df.loc[idx].sort_values("Year")
        cur = g["c"].values.astype(int)
        n = len(cur)
        forward = np.zeros_like(cur)
        for k in range(1, h + 1):
            shifted = np.zeros_like(cur)
            if k < n:
                shifted[:-k] = cur[k:]
            forward = np.logical_or(forward, shifted)
        df.loc[g.index, "crisis"] = forward.astype(int)

    return df


def prepare_dataset_for_h_and_group(df: pd.DataFrame, h: int, group: str) -> pd.DataFrame:
    df_h = add_forward_crisis_labels(df, h, crisis_col="crisis", country_col="WEOCountryCode")

    if group == "EME":
        df_h = df_h[df_h["GroupName"] == "EME"].copy()
    elif group == "LIC":
        df_h = df_h[df_h["GroupName"] == "LIC"].copy()

    df_h = df_h[df_h["c"] == 0].copy()
    max_year = df_h["Year"].max()
    df_h = df_h[df_h["Year"] <= max_year - h].copy()
    df_h = df_h.sort_values(["Year", "WEOCountryCode"]).reset_index(drop=True)
    return df_h


def _assert_cols(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


# ----------------------------
# Per-split winsorization (train-only bounds)
# ----------------------------
def winsorize_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trim: int = WINSORIZE_TRIM,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute 1st/99th percentile bounds from train, clip both train and test.
    Only numeric feature columns are winsorized (identifiers and target excluded).
    """
    train_out = train_df.copy()
    test_out = test_df.copy()

    num_cols = [
        c for c in train_df.columns
        if c not in WINSORIZE_EXCLUDE and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    lo = trim / 100.0
    hi = 1.0 - lo

    for col in num_cols:
        vals = train_out[col].dropna()
        if len(vals) == 0:
            continue
        lower = vals.quantile(lo)
        upper = vals.quantile(hi)
        if lower == upper:
            continue
        train_out[col] = train_out[col].clip(lower=lower, upper=upper)
        if col in test_out.columns:
            test_out[col] = test_out[col].clip(lower=lower, upper=upper)

    return train_out, test_out


# ----------------------------
# Per-split imputation (train-only fit, apply to train+test)
# ----------------------------
def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def impute_groupname_only(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    var: str,
    country_col: str = "WEOCountryCode",
    group_col: str = "GroupName",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tier-1 imputation (train-only fit):
      Train: country interpolation -> GroupName median -> global median
      Test:  GroupName median -> global median (no interpolation for single-year rows)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    if var not in train_df.columns:
        return train_df, test_df

    # 1. Country-level interpolation on train (MUST filter to train first)
    train_df[var] = train_df.groupby(country_col)[var].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )

    # 2. Compute GroupName medians from post-interpolation train
    group_medians = train_df.groupby(group_col)[var].median()

    # 3. Compute global median from post-interpolation train
    global_median = train_df[var].median()

    # 4. Fill remaining NaNs in train: GroupName median -> global median
    for gname, gmed in group_medians.items():
        mask = (train_df[group_col] == gname) & train_df[var].isna()
        train_df.loc[mask, var] = gmed
    train_df[var] = train_df[var].fillna(global_median)

    # 5. Fill test: GroupName median -> global median
    if var in test_df.columns:
        for gname, gmed in group_medians.items():
            mask = (test_df[group_col] == gname) & test_df[var].isna()
            test_df.loc[mask, var] = gmed
        test_df[var] = test_df[var].fillna(global_median)

    return train_df, test_df


def impute_groupname_year(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    var: str,
    country_col: str = "WEOCountryCode",
    group_col: str = "GroupName",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tier-2 imputation (train-only fit):
      Train: country interpolation -> GroupName x Year median -> Year median
      Test:  test year t is by construction always outside the train window
             (separated by at least h years), so GroupName x Year and Year
             medians for year t never exist in train.  Test imputation therefore
             always degrades to Tier-1-like fills: GroupName median -> global median.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    if var not in train_df.columns:
        return train_df, test_df

    # 1. Country-level interpolation on train
    train_df[var] = train_df.groupby(country_col)[var].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )

    # 2. Compute GroupName x Year medians from train
    gy_medians = train_df.groupby([group_col, "Year"])[var].median()

    # 3. Compute Year medians from train
    year_medians = train_df.groupby("Year")[var].median()

    # 4. Fill remaining NaNs in train: GroupName x Year -> Year median
    for (gname, yr), gmed in gy_medians.items():
        mask = (train_df[group_col] == gname) & (train_df["Year"] == yr) & train_df[var].isna()
        train_df.loc[mask, var] = gmed
    for yr, ymed in year_medians.items():
        mask = (train_df["Year"] == yr) & train_df[var].isna()
        train_df.loc[mask, var] = ymed

    # 5. For test: GroupName x Year medians won't match (test year outside train).
    #    Fall back to GroupName median (from train) -> global median (from train).
    group_medians = train_df.groupby(group_col)[var].median()
    global_median = train_df[var].median()

    if var in test_df.columns:
        # Try GroupName x Year first (in case test year happens to overlap -- rare but safe)
        for (gname, yr), gmed in gy_medians.items():
            mask = (test_df[group_col] == gname) & (test_df["Year"] == yr) & test_df[var].isna()
            test_df.loc[mask, var] = gmed
        # Then Year median
        for yr, ymed in year_medians.items():
            mask = (test_df["Year"] == yr) & test_df[var].isna()
            test_df.loc[mask, var] = ymed
        # Then GroupName median
        for gname, gmed in group_medians.items():
            mask = (test_df[group_col] == gname) & test_df[var].isna()
            test_df.loc[mask, var] = gmed
        # Then global median
        test_df[var] = test_df[var].fillna(global_median)

    return train_df, test_df


def impute_global_median(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    var: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tier-3 imputation: simple median from train, applied to both.
    Used for global variables (lnVIX, WGDPg, etc.).
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    if var not in train_df.columns:
        return train_df, test_df

    med = train_df[var].median()
    train_df[var] = train_df[var].fillna(med)
    if var in test_df.columns:
        test_df[var] = test_df[var].fillna(med)

    return train_df, test_df


def impute_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all imputation tiers on a single train/test split.
    Column groups depend on dataset_key (baseline vs expanded).
    """
    # Baseline columns (always applied)
    for var in BASELINE_GROUPNAME_ONLY:
        train_df, test_df = impute_groupname_only(train_df, test_df, var)
    for var in BASELINE_GROUPNAME_YEAR:
        train_df, test_df = impute_groupname_year(train_df, test_df, var)
    for var in BASELINE_GLOBAL_MEDIAN:
        train_df, test_df = impute_global_median(train_df, test_df, var)

    # Expanded-only columns (only when dataset is "expanded")
    if dataset_key == "expanded":
        for var in EXPANDED_GROUPNAME_ONLY:
            train_df, test_df = impute_groupname_only(train_df, test_df, var)
        for var in EXPANDED_GROUPNAME_YEAR:
            train_df, test_df = impute_groupname_year(train_df, test_df, var)

    # Special case: gee (WGI Governance Effectiveness) starts in 1996, so
    # early training windows (pre-1996) are 100% NaN and imputation produces
    # NaN medians.  Fill with 0 (the WGI scale midpoint = world average).
    if "gee" in train_df.columns:
        train_df["gee"] = train_df["gee"].fillna(0)
    if "gee" in test_df.columns:
        test_df["gee"] = test_df["gee"].fillna(0)

    return train_df, test_df


# ----------------------------
# Build train/test keys from reference horizon dataset (one per t)
# ----------------------------
def build_split_keys_per_t(
    ref_horizon: pd.DataFrame, h: int, group: str
) -> List[Tuple[int, int, int, pd.DataFrame, pd.DataFrame]]:
    """Returns list of (t, train_lo, train_hi, train_keys, test_keys)."""
    t_max = t_max_fn(h)
    out = []
    for t in range(T_MIN, t_max + 1):
        train_lo = t - h - LOOKBACK + 1
        train_hi = t - h
        train_keys = ref_horizon[
            (ref_horizon["Year"] >= train_lo) & (ref_horizon["Year"] <= train_hi)
        ][["WEOCountryCode", "Year"]].drop_duplicates()
        test_keys = ref_horizon[ref_horizon["Year"] == t][["WEOCountryCode", "Year"]].drop_duplicates()
        out.append((t, train_lo, train_hi, train_keys, test_keys))
    return out


# ----------------------------
# Write train/test for one dataset with per-split preprocessing
# ----------------------------
def write_splits_for_dataset(
    df_h: pd.DataFrame,
    dataset_key: str,
    h: int,
    group: str,
    split_keys: List[Tuple[int, int, int, pd.DataFrame, pd.DataFrame]],
    train_dir: Path,
    test_dir: Path,
) -> pd.DataFrame:
    manifest_rows = []
    for t, train_lo, train_hi, train_keys, test_keys in split_keys:
        train_df = df_h.merge(train_keys, on=["WEOCountryCode", "Year"], how="inner")
        test_df = df_h.merge(test_keys, on=["WEOCountryCode", "Year"], how="inner")

        if train_df.empty or test_df.empty:
            manifest_rows.append({
                "Dataset": dataset_key,
                "Horizon": h,
                "Group": group,
                "t": t,
                "TrainYearMin": train_lo,
                "TrainYearMax": train_hi,
                "TrainRows": len(train_df),
                "TestRows": len(test_df),
                "Saved": False,
                "Reason": "Empty train or test",
                "TrainPath": "",
                "TestPath": "",
                "TrainCountries": 0,
                "TestCountries": 0,
                "TrainCrisisRate": np.nan,
                "TestCrisisRate": np.nan,
            })
            continue

        # --- Per-split preprocessing (leak-free) ---
        # Step 1: Winsorize (bounds from train only)
        train_df, test_df = winsorize_split(train_df, test_df, WINSORIZE_TRIM)

        # Step 2: Impute (fit on train only, apply to test)
        train_df, test_df = impute_split(train_df, test_df, dataset_key)

        train_path = train_dir / f"train_{dataset_key}_h{h}_{group}_t{t}.csv"
        test_path = test_dir / f"test_{dataset_key}_h{h}_{group}_t{t}.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        manifest_rows.append({
            "Dataset": dataset_key,
            "Horizon": h,
            "Group": group,
            "t": t,
            "TrainYearMin": train_lo,
            "TrainYearMax": train_hi,
            "TrainRows": len(train_df),
            "TestRows": len(test_df),
            "TrainCountries": train_df["WEOCountryCode"].nunique(),
            "TestCountries": test_df["WEOCountryCode"].nunique(),
            "TrainCrisisRate": float(train_df["crisis"].mean()),
            "TestCrisisRate": float(test_df["crisis"].mean()),
            "TrainPath": str(train_path),
            "TestPath": str(test_path),
            "Saved": True,
            "Reason": "",
        })

    return pd.DataFrame(manifest_rows)


def main() -> None:
    print("=" * 70)
    print("Make splits: horizon preparation + per-split winsorize/impute (data/final/)")
    print("=" * 70)

    horizonsplit_dir = OUT_BASE / "horizonsplit"
    train_dir = OUT_BASE / "train"
    test_dir = OUT_BASE / "test"
    manifest_dir = OUT_BASE / "manifest"
    for d in (OUT_BASE, horizonsplit_dir, train_dir, test_dir, manifest_dir):
        d.mkdir(parents=True, exist_ok=True)

    for h in HORIZONS:
        for group in GROUPS:
            print(f"\n--- Horizon h={h}, group={group} ---")

            # Reference: first dataset defines (WEOCountryCode, Year) for train/test
            ref_file = DATASET_FILES[DATASET_KEYS[0]]
            ref_path = CONSTRUCTED_DIR / ref_file
            if not ref_path.exists():
                raise FileNotFoundError(f"Reference dataset not found: {ref_path}")
            ref_df = pd.read_csv(ref_path)
            _assert_cols(ref_df, ref_path)
            ref_df["crisis"] = ref_df["crisis"].fillna(0).astype(int)
            ref_df["WEOCountryCode"] = ref_df["WEOCountryCode"].astype(str)
            ref_df["Year"] = pd.to_numeric(ref_df["Year"], errors="coerce")

            ref_horizon = prepare_dataset_for_h_and_group(ref_df, h, group)
            split_keys = build_split_keys_per_t(ref_horizon, h, group)

            for dataset_key in DATASET_KEYS:
                csv_file = DATASET_FILES[dataset_key]
                path = CONSTRUCTED_DIR / csv_file
                if not path.exists():
                    print(f"  Skip (missing): {path}")
                    continue
                df = pd.read_csv(path)
                _assert_cols(df, path)
                df["crisis"] = df["crisis"].fillna(0).astype(int)
                df["WEOCountryCode"] = df["WEOCountryCode"].astype(str)
                df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

                df_h = prepare_dataset_for_h_and_group(df, h, group)

                # Save raw horizonsplit (unwinsorized, unimputed)
                horizonsplit_path = horizonsplit_dir / f"{dataset_key}_h{h}_{group}.csv"
                df_h.to_csv(horizonsplit_path, index=False)

                # Write per-split winsorized+imputed train/test
                manifest_df = write_splits_for_dataset(
                    df_h, dataset_key, h, group, split_keys, train_dir, test_dir
                )
                manifest_path = manifest_dir / f"manifest_{dataset_key}_h{h}_{group}.csv"
                manifest_df.to_csv(manifest_path, index=False)

                saved_n = int(manifest_df["Saved"].sum())
                print(f"  {dataset_key}: horizonsplit + {saved_n}/{len(manifest_df)} splits written")

    # Summary
    train_files = list(train_dir.glob("*.csv"))
    test_files = list(test_dir.glob("*.csv"))
    print("\n" + "=" * 70)
    print(f"Output base: {OUT_BASE}")
    print(f"  horizonsplit: {len(list(horizonsplit_dir.glob('*.csv')))} files")
    print(f"  train: {len(train_files)} files")
    print(f"  test: {len(test_files)} files")
    print(f"  manifest: {len(list(manifest_dir.glob('*.csv')))} files")
    print("Done.")


if __name__ == "__main__":
    main()
