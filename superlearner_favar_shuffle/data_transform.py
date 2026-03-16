# ================================================================
# data_transform.py (Superlearner FAVAR-Net v2.2 - Shuffle)
# ================================================================
#
# Reads existing train/test CSVs from data/final, engineers 18
# high-signal features, and writes augmented CSVs to
# traintest_transforms/.
#
# Changes vs v2.1 (superlearner_favar/data_transform.py):
#   1. build_favar_inputs() -- expanded THEORY_INTERACTIONS (14 terms)
#      with momentum, momentum interactions, and second-order terms
#   2. compute_oof_neural_risk_score() -- shuffled inner val split
#      with M=INNER_REPEATS averaged repeats for stable OOF
#   3. Final FAVAR-Net training -- shuffled val split (not last 10%)
#
# Feature engineering per split year t:
#   1. Load train/test CSVs (winsorized, imputed)
#   2. Load horizonsplit CSV (for previous-year lookups)
#   3. Compute 5 momentum features (delta_X = X_y - X_{y-1})
#   4. Compute 3 peer deviation features (X - median_group)
#   5. Compute 3 interaction features (D*r_g, ED/FXR, CA/FXR)
#   5b. Compute 6 advanced features (triple interaction, LIC,
#       DSED/FXR, debt acceleration, CA*VIX)
#   6. Compute global factors (median+IQR -> PCA d=2) from train
#   7. Build FAVAR-Net inputs (lags + factors + interactions)
#   8. Train FAVAR-Net classifier with OUT-OF-FOLD predictions
#      on training data (10-fold by country) -> neural_risk_score
#   9. Train final FAVAR-Net on all training data -> predict test
#  10. Save interpretability artifacts
#  11. Augment and save CSVs
#
# CLI:
#   python -m superlearner_favar_shuffle.data_transform \
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
from sklearn.decomposition import PCA

from superlearner_favar_shuffle.config import (
    SPLITS_BASE,
    TRANSFORMS_DIR,
    RESULTS_DIR,
    VALID_DATASETS,
    VALID_HORIZONS,
    VALID_GROUPS,
    FORECAST_VARS,
    FAVAR_CONFIG,
    THEORY_INTERACTIONS,
    INNER_REPEATS,
    NON_FEATURE_COLS,
    TARGET_COL,
    COUNTRY_COL,
    YEAR_COL,
    GROUP_COL,
    T_MIN,
    MOMENTUM_VARS,
    PEER_DEV_VARS,
    INTERACTION_FEATURES,
    ADVANCED_FEATURES,
    ENGINEERED_FEATURE_NAMES,
    get_t_max,
)

warnings.filterwarnings("ignore")


# ================================================================
# Helper: discover years, load split
# ================================================================

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


def load_horizonsplit(dataset: str, horizon: int, group: str):
    """Load the horizonsplit CSV (unwinsorized, unimputed, c=0 only)."""
    path = os.path.join(SPLITS_BASE, "horizonsplit", f"{dataset}_h{horizon}_{group}.csv")
    return pd.read_csv(path)


# ================================================================
# Feature Engineering: Momentum
# ================================================================

def engineer_momentum_features(df: pd.DataFrame, horizonsplit: pd.DataFrame) -> pd.DataFrame:
    """
    Add year-over-year change features: delta_X = X_y - X_{y-1}.

    Previous-year values are looked up from the horizonsplit (which has
    the full time range), keyed by (country, year-1).

    Args:
        df: train_df or test_df (winsorized, imputed)
        horizonsplit: full horizonsplit data

    Returns:
        df with 5 new columns: delta_D, delta_FXR, delta_lnGDPpc, delta_r_g, delta_CA
    """
    df = df.copy()

    # Build lookup: {(country, year): {var: value}} from horizonsplit
    prev_lookup = {}
    for _, row in horizonsplit.iterrows():
        key = (str(row[COUNTRY_COL]), int(row[YEAR_COL]))
        vals = {}
        for v in MOMENTUM_VARS:
            if v in row.index:
                vals[v] = row[v]
        prev_lookup[key] = vals

    for v in MOMENTUM_VARS:
        col_name = f"delta_{v}"
        deltas = np.zeros(len(df), dtype=np.float32)
        for i, (idx, row) in enumerate(df.iterrows()):
            country = str(row[COUNTRY_COL])
            year = int(row[YEAR_COL])
            current_val = row[v] if v in row.index and pd.notna(row[v]) else 0.0
            prev_key = (country, year - 1)
            if prev_key in prev_lookup and v in prev_lookup[prev_key]:
                prev_val = prev_lookup[prev_key][v]
                if pd.notna(prev_val):
                    deltas[i] = float(current_val) - float(prev_val)
                # else: delta stays 0 (missing previous year)
            # else: delta stays 0 (no previous year available)
        df[col_name] = deltas

    return df


# ================================================================
# Feature Engineering: Peer Deviation
# ================================================================

def compute_group_medians(train_df: pd.DataFrame) -> dict:
    """
    Compute GroupName x Year medians for peer deviation variables.

    Returns:
        medians: {(GroupName, Year): {var: median_value}}
    """
    medians = {}
    for (grp, yr), sub in train_df.groupby([GROUP_COL, YEAR_COL]):
        med = {}
        for v in PEER_DEV_VARS:
            if v in sub.columns:
                med[v] = sub[v].median()
        medians[(grp, int(yr))] = med
    return medians


def engineer_peer_deviation_features(
    df: pd.DataFrame, group_medians: dict, fallback_medians: dict = None,
) -> pd.DataFrame:
    """
    Add peer deviation features: X_dev = X_country - median(X, GroupName, Year).

    Medians are pre-computed from training data only.
    For test rows where the year is not in the train medians, we use
    the closest available train year's median.
    """
    df = df.copy()

    # Build sorted list of available years per group for fallback
    group_years = {}
    for (grp, yr) in group_medians:
        group_years.setdefault(grp, []).append(yr)
    for grp in group_years:
        group_years[grp] = sorted(group_years[grp])

    def _get_median(grp, yr, var):
        key = (grp, yr)
        if key in group_medians and var in group_medians[key]:
            val = group_medians[key][var]
            if pd.notna(val):
                return val
        # Fallback: nearest train year for this group
        if grp in group_years:
            years_list = group_years[grp]
            # Find closest year
            closest = min(years_list, key=lambda y: abs(y - yr))
            fkey = (grp, closest)
            if fkey in group_medians and var in group_medians[fkey]:
                val = group_medians[fkey][var]
                if pd.notna(val):
                    return val
        return 0.0  # last resort

    for v in PEER_DEV_VARS:
        col_name = f"{v}_dev"
        devs = np.zeros(len(df), dtype=np.float32)
        for i, (idx, row) in enumerate(df.iterrows()):
            grp = row[GROUP_COL]
            yr = int(row[YEAR_COL])
            country_val = row[v] if v in row.index and pd.notna(row[v]) else 0.0
            med = _get_median(grp, yr, v)
            devs[i] = float(country_val) - float(med)
        df[col_name] = devs

    return df


# ================================================================
# Feature Engineering: Interactions
# ================================================================

def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add theory-driven interaction features:
      D_x_r_g    = D * r_g       (debt service burden)
      ED_div_FXR = ED / max(FXR, 0.01)  (external vulnerability)
      CA_div_FXR = CA / max(FXR, 0.01)  (external balance vs reserves)
    """
    df = df.copy()

    D = df["D"].fillna(0.0).values.astype(np.float64)
    r_g = df["r_g"].fillna(0.0).values.astype(np.float64)
    ED = df["ED"].fillna(0.0).values.astype(np.float64)
    FXR = df["FXR"].fillna(0.0).values.astype(np.float64)
    CA = df["CA"].fillna(0.0).values.astype(np.float64)

    df["D_x_r_g"] = (D * r_g).astype(np.float32)
    df["ED_div_FXR"] = (ED / np.maximum(FXR, 0.01)).astype(np.float32)
    df["CA_div_FXR"] = (CA / np.maximum(FXR, 0.01)).astype(np.float32)

    return df


# ================================================================
# Feature Engineering: Advanced Features (v2.1)
# ================================================================

def engineer_advanced_features(
    df: pd.DataFrame, horizonsplit: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add 6 advanced features derived from data analysis:

      D_x_r_g_div_FXR  = D * max(r_g, 0) / max(FXR, 0.5)
                          Triple interaction: debt sustainability pressure
                          when reserves are low

      is_LIC            = 1 if GroupName == "LIC", else 0
                          LICs have 1.7x higher crisis rate

      is_LIC_x_D        = is_LIC * D
                          Debt is more dangerous for LICs

      DSED_div_FXR      = DSED / max(FXR, 0.5)
                          Short-term rollover risk

      delta_delta_D      = (D_y - D_{y-1}) - (D_{y-1} - D_{y-2})
                          Debt acceleration (2nd derivative)

      CA_x_lnVIX        = CA * lnVIX
                          CA deficits more dangerous in high-volatility regimes
    """
    df = df.copy()

    D = df["D"].fillna(0.0).values.astype(np.float64)
    r_g = df["r_g"].fillna(0.0).values.astype(np.float64)
    FXR = df["FXR"].fillna(0.0).values.astype(np.float64)
    ED = df["ED"].fillna(0.0).values.astype(np.float64)
    CA = df["CA"].fillna(0.0).values.astype(np.float64)
    DSED = df["DSED"].fillna(0.0).values.astype(np.float64) if "DSED" in df.columns else np.zeros(len(df))
    lnVIX = df["lnVIX"].fillna(0.0).values.astype(np.float64) if "lnVIX" in df.columns else np.zeros(len(df))

    # 1) Triple interaction: D * max(r_g, 0) / max(FXR, 0.5)
    df["D_x_r_g_div_FXR"] = (D * np.maximum(r_g, 0.0) / np.maximum(FXR, 0.5)).astype(np.float32)

    # 2) LIC binary indicator
    df["is_LIC"] = (df[GROUP_COL] == "LIC").astype(np.float32)

    # 3) Group-conditional debt: is_LIC * D
    df["is_LIC_x_D"] = (df["is_LIC"].values * D).astype(np.float32)

    # 4) Short-term rollover risk: DSED / max(FXR, 0.5)
    df["DSED_div_FXR"] = (DSED / np.maximum(FXR, 0.5)).astype(np.float32)

    # 5) Debt acceleration: delta_delta_D = (D_y - D_{y-1}) - (D_{y-1} - D_{y-2})
    #    Requires looking up D at y-1 and y-2 from horizonsplit
    prev_lookup = {}
    for _, row in horizonsplit.iterrows():
        key = (str(row[COUNTRY_COL]), int(row[YEAR_COL]))
        if "D" in row.index and pd.notna(row["D"]):
            prev_lookup[key] = float(row["D"])

    accel = np.zeros(len(df), dtype=np.float32)
    for i, (idx, row) in enumerate(df.iterrows()):
        country = str(row[COUNTRY_COL])
        year = int(row[YEAR_COL])
        d_curr = float(row["D"]) if pd.notna(row["D"]) else 0.0
        d_m1 = prev_lookup.get((country, year - 1), None)
        d_m2 = prev_lookup.get((country, year - 2), None)
        if d_m1 is not None and d_m2 is not None:
            delta_curr = d_curr - d_m1
            delta_prev = d_m1 - d_m2
            accel[i] = delta_curr - delta_prev
        # else: stays 0
    df["delta_delta_D"] = accel

    # 6) CA deficit x global volatility
    df["CA_x_lnVIX"] = (CA * lnVIX).astype(np.float32)

    return df


# ================================================================
# Global Factor Construction (median + IQR -> PCA)
# ================================================================

def compute_global_factors(train_df, test_df, forecast_vars, n_factors, random_state=1234):
    """
    Build global latent factors from cross-sectional macro summaries.

    For each year y in train_df:
      - Compute median and IQR of each FORECAST_VAR across countries
      - Stack into G_y vector of length 2*V
      - Fit PCA(n_factors) on train years only
      - Transform all years (train + test)

    Returns:
        factors_by_year: dict mapping year -> np.array of shape (n_factors,)
        pca, g_mean, g_std: for reproducibility
    """
    V = len(forecast_vars)

    train_years_sorted = sorted(train_df[YEAR_COL].unique())
    G_rows = []
    G_years = []
    for y in train_years_sorted:
        year_data = train_df[train_df[YEAR_COL] == y][forecast_vars]
        if len(year_data) < 2:
            continue
        medians = year_data.median().values
        q75 = year_data.quantile(0.75).values
        q25 = year_data.quantile(0.25).values
        iqr = q75 - q25
        g_y = np.concatenate([medians, iqr])
        G_rows.append(g_y)
        G_years.append(y)

    G_matrix = np.array(G_rows, dtype=np.float32)

    # Handle NaN
    col_means = np.nanmean(G_matrix, axis=0)
    nans = np.where(np.isnan(G_matrix))
    if len(nans[0]) > 0:
        G_matrix[nans] = np.take(col_means, nans[1])

    # Standardize
    g_mean = G_matrix.mean(axis=0)
    g_std = G_matrix.std(axis=0)
    g_std[g_std < 1e-8] = 1.0
    G_norm = (G_matrix - g_mean) / g_std

    # PCA
    n_comp = min(n_factors, len(G_norm), G_norm.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    pca.fit(G_norm)

    train_factors = pca.transform(G_norm)
    factors_by_year = {}
    for i, y in enumerate(G_years):
        factors_by_year[y] = train_factors[i]

    # Transform test years
    test_years = sorted(test_df[YEAR_COL].unique())
    for y in test_years:
        if y in factors_by_year:
            continue
        year_data = test_df[test_df[YEAR_COL] == y][forecast_vars]
        if len(year_data) < 2:
            factors_by_year[y] = np.zeros(n_comp, dtype=np.float32)
            continue
        medians = year_data.median().values
        q75 = year_data.quantile(0.75).values
        q25 = year_data.quantile(0.25).values
        iqr = q75 - q25
        g_y = np.concatenate([medians, iqr]).astype(np.float32)
        nan_mask = np.isnan(g_y)
        if nan_mask.any():
            g_y[nan_mask] = col_means[nan_mask]
        g_y_norm = (g_y - g_mean) / g_std
        factors_by_year[y] = pca.transform(g_y_norm.reshape(1, -1))[0]

    var_explained = pca.explained_variance_ratio_
    print(f"  Global factors (median+IQR PCA): {n_comp} components, "
          f"variance explained: {[f'{v:.1%}' for v in var_explained]}")

    return factors_by_year, pca, g_mean, g_std


# ================================================================
# Build FAVAR Inputs (flat lag vectors + global factors + interactions)
# ================================================================

def build_favar_inputs(
    df: pd.DataFrame,
    horizonsplit: pd.DataFrame,
    factors_by_year: dict,
    forecast_vars: list,
    p_lags: int,
    n_factors: int,
):
    """
    Build FAVAR-Net input matrices for each row in df.

    Lags are looked up from the horizonsplit (which has the full time range
    of c=0 rows), not from df itself. This allows test rows to access
    years outside the training window for lag construction.

    For each country-year row at year y:
      a) Local lags: [x_{y-1}, ..., x_{y-p}] flattened -> (p * V,)
      b) Global factor lags: [f_y, f_{y-1}, ..., f_{y-p+1}] -> (p * d,)
      c) Theory interactions at year y -> (n_interactions,)
      d) f_current = f_y -> (d,) separate output for regime gating

    v2.2: Expanded THEORY_INTERACTIONS (14 terms including momentum,
          momentum interactions, second-order, and conditional M2_GDP).

    Returns:
        Z: (n_rows, input_dim)
        f_current: (n_rows, n_factors)
    """
    V = len(forecast_vars)

    # Build history lookup from horizonsplit: {country: {year: np.array(V,)}}
    history = {}
    for _, row in horizonsplit.iterrows():
        country = str(row[COUNTRY_COL])
        y = int(row[YEAR_COL])
        vals = np.array([row[v] if v in row.index else np.nan for v in forecast_vars], dtype=np.float32)
        if country not in history:
            history[country] = {}
        history[country][y] = vals

    # Build extra history for variables NOT in forecast_vars (CA, M2_GDP)
    extra_vars = ["CA", "M2_GDP"]
    extra_history = {}  # {country: {year: {var: value}}}
    for _, row in horizonsplit.iterrows():
        country = str(row[COUNTRY_COL])
        y = int(row[YEAR_COL])
        ev = {}
        for v in extra_vars:
            if v in row.index and pd.notna(row[v]):
                ev[v] = float(row[v])
        if country not in extra_history:
            extra_history[country] = {}
        extra_history[country][y] = ev

    n_rows = len(df)
    local_lag_dim = p_lags * V
    global_lag_dim = p_lags * n_factors
    n_interactions = len(THEORY_INTERACTIONS)
    input_dim = local_lag_dim + global_lag_dim + n_interactions

    Z = np.zeros((n_rows, input_dim), dtype=np.float32)
    f_current = np.zeros((n_rows, n_factors), dtype=np.float32)

    var_idx = {v: i for i, v in enumerate(forecast_vars)}

    for row_i, (idx, row) in enumerate(df.iterrows()):
        country = str(row[COUNTRY_COL])
        y = int(row[YEAR_COL])
        ch = history.get(country, {})
        ch_extra = extra_history.get(country, {})

        # a) Local lags
        offset = 0
        for lag in range(1, p_lags + 1):
            lag_year = y - lag
            if lag_year in ch:
                Z[row_i, offset:offset + V] = ch[lag_year]
            offset += V

        # b) Global factor lags
        for lag in range(0, p_lags):
            lag_year = y - lag
            if lag_year in factors_by_year:
                f_vec = factors_by_year[lag_year]
                Z[row_i, offset:offset + n_factors] = f_vec[:n_factors]
            offset += n_factors

        # c) Theory interactions at year y
        current_vals = np.array([row[v] if v in row.index else 0.0 for v in forecast_vars], dtype=np.float32)
        current_vals = np.nan_to_num(current_vals, nan=0.0)

        f_y = factors_by_year.get(y, np.zeros(n_factors, dtype=np.float32))
        f_current[row_i] = f_y[:n_factors]

        prev_vals = ch.get(y - 1, np.zeros(V, dtype=np.float32))
        prev_vals = np.nan_to_num(prev_vals, nan=0.0)

        # Pre-compute commonly needed deltas for interactions
        delta_D_val = current_vals[var_idx.get("D", 0)] - prev_vals[var_idx.get("D", 0)]
        delta_FXR_val = current_vals[var_idx.get("FXR", 0)] - prev_vals[var_idx.get("FXR", 0)]

        for inter_i, inter_name in enumerate(THEORY_INTERACTIONS):
            val = 0.0

            # --- Original 6 interactions (unchanged) ---
            if inter_name == "D_x_r_g":
                val = current_vals[var_idx.get("D", 0)] * current_vals[var_idx.get("r_g", 0)]
            elif inter_name == "D_x_GlobalPC1":
                val = current_vals[var_idx.get("D", 0)] * (f_y[0] if len(f_y) > 0 else 0.0)
            elif inter_name == "CA_x_FXR":
                val = current_vals[var_idx.get("CA", 0)] * current_vals[var_idx.get("FXR", 0)]
            elif inter_name == "r_g_x_GlobalPC1":
                val = current_vals[var_idx.get("r_g", 0)] * (f_y[0] if len(f_y) > 0 else 0.0)
            elif inter_name == "ED_x_FXR":
                val = current_vals[var_idx.get("ED", 0)] * current_vals[var_idx.get("FXR", 0)]
            elif inter_name == "delta_D":
                val = delta_D_val

            # --- New first-order momentum (4) ---
            elif inter_name == "delta_FXR":
                val = delta_FXR_val
            elif inter_name == "delta_ED":
                val = current_vals[var_idx.get("ED", 0)] - prev_vals[var_idx.get("ED", 0)]
            elif inter_name == "delta_r_g":
                val = current_vals[var_idx.get("r_g", 0)] - prev_vals[var_idx.get("r_g", 0)]
            elif inter_name == "delta_CA":
                # CA may not be in forecast_vars, read from row + extra_history
                ca_curr = float(row["CA"]) if "CA" in row.index and pd.notna(row["CA"]) else 0.0
                ca_prev_dict = ch_extra.get(y - 1, {})
                ca_prev = ca_prev_dict.get("CA", 0.0)
                val = ca_curr - ca_prev

            # --- New momentum interactions (2) ---
            elif inter_name == "delta_D_x_r_g":
                val = delta_D_val * current_vals[var_idx.get("r_g", 0)]
            elif inter_name == "delta_D_x_delta_FXR":
                val = delta_D_val * delta_FXR_val

            # --- New second-order (1) ---
            elif inter_name == "delta_delta_D":
                # (D_y - D_{y-1}) - (D_{y-1} - D_{y-2})
                prev_vals_m2 = ch.get(y - 2, np.zeros(V, dtype=np.float32))
                prev_vals_m2 = np.nan_to_num(prev_vals_m2, nan=0.0)
                d_idx = var_idx.get("D", 0)
                delta_curr = current_vals[d_idx] - prev_vals[d_idx]
                delta_prev = prev_vals[d_idx] - prev_vals_m2[d_idx]
                val = delta_curr - delta_prev

            # --- New conditional: M2_GDP momentum (expanded dataset only) ---
            elif inter_name == "delta_M2_GDP":
                # Gracefully skip if M2_GDP column is absent
                if "M2_GDP" in row.index and pd.notna(row["M2_GDP"]):
                    m2_curr = float(row["M2_GDP"])
                    m2_prev_dict = ch_extra.get(y - 1, {})
                    m2_prev = m2_prev_dict.get("M2_GDP", 0.0)
                    val = m2_curr - m2_prev
                else:
                    val = 0.0

            Z[row_i, local_lag_dim + global_lag_dim + inter_i] = val

    # Replace any NaN/Inf
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    f_current = np.nan_to_num(f_current, nan=0.0)

    return Z, f_current


# ================================================================
# Save FAVAR Interpretability Artifacts
# ================================================================

def save_favar_artifacts(forecaster, Z, f, save_dir, year):
    """Save regime weights and expert coefficients for interpretability."""
    os.makedirs(save_dir, exist_ok=True)

    try:
        regime = forecaster.extract_regime_weights(Z, f)
        np.savez_compressed(
            os.path.join(save_dir, f"regime_weights_t{year}.npz"),
            pi=regime["pi"],
            mean_pi=regime["mean_pi"],
            f_values=regime["f_values"],
        )
        print(f"    Regime weights: mean_pi = {regime['mean_pi']}")

        coeffs = forecaster.extract_expert_coefficients()
        np.savez_compressed(
            os.path.join(save_dir, f"expert_coefficients_t{year}.npz"),
            **coeffs,
        )
        print(f"    Expert coefficients saved (W_0: {coeffs['W_0'].shape}, W_1: {coeffs['W_1'].shape})")

    except Exception as e:
        print(f"    WARNING: Could not extract FAVAR artifacts: {e}")


# ================================================================
# Out-of-Fold Neural Risk Score (v2.2: shuffled val + M=3 averaging)
# ================================================================

def compute_oof_neural_risk_score(
    train_df: pd.DataFrame,
    horizonsplit: pd.DataFrame,
    factors_by_year: dict,
    forecast_vars: list,
    p_lags: int,
    n_factors: int,
    n_folds: int,
    random_state: int,
    inner_repeats: int = INNER_REPEATS,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute out-of-fold neural_risk_score for training data.

    Splits countries into K folds. For each fold:
      - Train FAVAR-Net classifier on countries NOT in the fold
      - Predict on countries IN the fold
    This ensures no label leakage from NN to RF.

    v2.2 change: Instead of using the last 10% as validation (biased),
    we use M=inner_repeats random shuffled validation splits. Each
    repeat trains a separate FAVAR-Net with a different random val
    split, and the held-out predictions are averaged across repeats
    for stability.

    Returns:
        oof_scores: (n_train,) array of crisis probabilities
    """
    from superlearner_favar_shuffle.models.favar_net import FAVARForecaster

    countries = train_df[COUNTRY_COL].values
    unique_countries = np.unique(countries)
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_countries)
    folds = np.array_split(unique_countries, n_folds)

    oof_scores = np.zeros(len(train_df), dtype=np.float32)

    # Build FAVAR inputs for ALL training rows once
    Z_all, f_all = build_favar_inputs(
        train_df, horizonsplit, factors_by_year, forecast_vars, p_lags, n_factors,
    )
    y_all = train_df[TARGET_COL].values.astype(np.float32)

    for fold_k, fold_countries in enumerate(folds):
        fold_set = set(fold_countries)
        held_mask = np.array([c in fold_set for c in countries])
        train_mask = ~held_mask

        if verbose:
            n_tr = train_mask.sum()
            n_held = held_mask.sum()
            print(f"    OOF fold {fold_k+1}/{n_folds}: train={n_tr}, held-out={n_held}")

        Z_tr = Z_all[train_mask]
        f_tr = f_all[train_mask]
        y_tr = y_all[train_mask]

        Z_held = Z_all[held_mask]
        f_held = f_all[held_mask]

        if len(np.unique(y_tr)) < 2:
            if verbose:
                print(f"      Skipping fold {fold_k+1} (single class in train)")
            oof_scores[held_mask] = y_all[train_mask].mean()
            continue

        # v2.2: Shuffled val split with M=inner_repeats averaging
        n_val = max(1, int(0.1 * len(Z_tr)))
        inner_preds = []
        for m in range(inner_repeats):
            perm = np.random.RandomState(random_state + fold_k * 100 + m).permutation(len(Z_tr))
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            Z_val_m = Z_tr[val_idx]
            f_val_m = f_tr[val_idx]
            y_val_m = y_tr[val_idx]
            Z_tr_fit = Z_tr[tr_idx]
            f_tr_fit = f_tr[tr_idx]
            y_tr_fit = y_tr[tr_idx]

            if len(Z_tr_fit) < 5:
                Z_tr_fit, f_tr_fit, y_tr_fit = Z_tr, f_tr, y_tr
                Z_val_m, f_val_m, y_val_m = None, None, None

            favar_fold = FAVARForecaster(task="classify")
            favar_fold.fit(Z_tr_fit, y_tr_fit, f_tr_fit, Z_val_m, y_val_m, f_val_m, verbose=False)
            inner_preds.append(favar_fold.generate_neural_features(Z_held, f_held))

        oof_scores[held_mask] = np.mean(inner_preds, axis=0)

    return oof_scores


# ================================================================
# Main Transform Logic
# ================================================================

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

    # Lazy import so TF is only loaded when this module runs
    from superlearner_favar_shuffle.models.favar_net import FAVARForecaster

    # Load horizonsplit once (used for momentum feature lookups and lag construction)
    horizonsplit = load_horizonsplit(dataset, horizon, group)
    print(f"  Horizonsplit loaded: {len(horizonsplit)} rows")

    # Weights dir for interpretability artifacts
    run_id = f"superlearner_{dataset}_h{horizon}_{group}"
    weights_dir = os.path.join(RESULTS_DIR, run_id, "favar_weights")

    p_lags = FAVAR_CONFIG["p_lags"]
    n_factors = FAVAR_CONFIG["n_global_factors"]
    n_oof_folds = FAVAR_CONFIG["oof_folds"]
    forecast_vars = FORECAST_VARS

    manifest_rows = []

    for year in years:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  Year {year}  ({dataset} h{horizon} {group})")
        print(f"{'='*60}")

        train_df, test_df = load_split(dataset, horizon, group, year)
        print(f"  train rows={len(train_df)}, test rows={len(test_df)}")

        # ---- 1. Engineer momentum features ----
        print("  Engineering momentum features...")
        train_df = engineer_momentum_features(train_df, horizonsplit)
        test_df = engineer_momentum_features(test_df, horizonsplit)

        # ---- 2. Engineer peer deviation features ----
        print("  Engineering peer deviation features...")
        group_medians = compute_group_medians(train_df)
        train_df = engineer_peer_deviation_features(train_df, group_medians)
        test_df = engineer_peer_deviation_features(test_df, group_medians)

        # ---- 3. Engineer interaction features ----
        print("  Engineering interaction features...")
        train_df = engineer_interaction_features(train_df)
        test_df = engineer_interaction_features(test_df)

        # ---- 3b. Engineer advanced features (v2.1) ----
        print("  Engineering advanced features...")
        train_df = engineer_advanced_features(train_df, horizonsplit)
        test_df = engineer_advanced_features(test_df, horizonsplit)

        # ---- 4. Compute global factors (train only) ----
        factors_by_year, pca, g_mean, g_std = compute_global_factors(
            train_df, test_df, forecast_vars, n_factors,
            random_state=FAVAR_CONFIG["random_state"],
        )

        # ---- 5. Out-of-fold neural_risk_score for train ----
        print(f"  Computing OOF neural_risk_score ({n_oof_folds} folds, "
              f"{INNER_REPEATS} inner repeats)...")
        oof_scores = compute_oof_neural_risk_score(
            train_df, horizonsplit, factors_by_year,
            forecast_vars, p_lags, n_factors,
            n_folds=n_oof_folds,
            random_state=FAVAR_CONFIG["random_state"],
            inner_repeats=INNER_REPEATS,
            verbose=True,
        )
        train_df["neural_risk_score"] = oof_scores.astype(np.float32)
        print(f"  OOF neural_risk_score: mean={oof_scores.mean():.4f}, "
              f"std={oof_scores.std():.4f}")

        # ---- 6. Train final FAVAR-Net on ALL train data for test ----
        # v2.2: Use shuffled val split (not last 10%) with averaging
        print(f"  Training final FAVAR-Net classifier on all train data "
              f"({INNER_REPEATS} shuffled repeats)...")
        Z_train_all, f_train_all = build_favar_inputs(
            train_df, horizonsplit, factors_by_year, forecast_vars, p_lags, n_factors,
        )
        y_train_all = train_df[TARGET_COL].values.astype(np.float32)

        Z_test, f_test = build_favar_inputs(
            test_df, horizonsplit, factors_by_year, forecast_vars, p_lags, n_factors,
        )

        # Average test predictions over INNER_REPEATS shuffled val splits
        n_val = max(1, int(0.1 * len(Z_train_all)))
        test_preds_list = []
        favar_final = None  # keep last one for artifacts
        for m in range(INNER_REPEATS):
            perm = np.random.RandomState(FAVAR_CONFIG["random_state"] + 9999 + m).permutation(len(Z_train_all))
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            Z_val = Z_train_all[val_idx]
            f_val = f_train_all[val_idx]
            y_val = y_train_all[val_idx]
            Z_tr = Z_train_all[tr_idx]
            f_tr = f_train_all[tr_idx]
            y_tr = y_train_all[tr_idx]

            if len(Z_tr) < 5:
                Z_tr, f_tr, y_tr = Z_train_all, f_train_all, y_train_all
                Z_val, f_val, y_val = None, None, None

            favar_m = FAVARForecaster(task="classify")
            favar_m.fit(Z_tr, y_tr, f_tr, Z_val, y_val, f_val, verbose=(m == 0))
            test_preds_list.append(favar_m.generate_neural_features(Z_test, f_test))
            favar_final = favar_m  # keep last for artifacts

        test_neural_score = np.mean(test_preds_list, axis=0)
        test_df["neural_risk_score"] = test_neural_score.astype(np.float32)
        print(f"  Test neural_risk_score: mean={test_neural_score.mean():.4f}, "
              f"std={test_neural_score.std():.4f}")

        # ---- 7. Save FAVAR interpretability artifacts ----
        print("  Saving FAVAR interpretability artifacts...")
        max_samples = min(500, len(Z_train_all))
        save_favar_artifacts(
            favar_final, Z_train_all[:max_samples], f_train_all[:max_samples],
            weights_dir, year,
        )

        # ---- 8. Verify all engineered features are present ----
        for feat in ENGINEERED_FEATURE_NAMES:
            if feat not in train_df.columns:
                print(f"  WARNING: missing feature {feat} in train_df, adding zeros")
                train_df[feat] = 0.0
            if feat not in test_df.columns:
                print(f"  WARNING: missing feature {feat} in test_df, adding zeros")
                test_df[feat] = 0.0

        # ---- 9. Save augmented CSVs ----
        train_out = os.path.join(out_train_dir, f"train_{dataset}_h{horizon}_{group}_t{year}.csv")
        test_out = os.path.join(out_test_dir, f"test_{dataset}_h{horizon}_{group}_t{year}.csv")
        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)

        elapsed = time.time() - t0
        print(f"  Saved augmented CSVs ({elapsed:.1f}s)")

        manifest_rows.append({
            "Dataset": dataset, "Horizon": horizon, "Group": group,
            "t": year, "TrainRows": len(train_df), "TestRows": len(test_df),
            "TrainCrisisRate": train_df[TARGET_COL].mean(),
            "TestCrisisRate": test_df[TARGET_COL].mean(),
            "TrainPath": train_out, "TestPath": test_out,
        })

    # Write manifest
    manifest_path = os.path.join(out_manifest_dir, f"manifest_{dataset}_h{horizon}_{group}.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nManifest: {manifest_path}")
    print("Transform complete.")


# ================================================================
# CLI
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform data/final splits into FAVAR-Net v2.2 (shuffle) feature-augmented CSVs"
    )
    parser.add_argument("--dataset", type=str, choices=VALID_DATASETS, required=True)
    parser.add_argument("--horizon", type=int, choices=VALID_HORIZONS, required=True)
    parser.add_argument("--group", type=str, choices=VALID_GROUPS, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Superlearner FAVAR-Net v2.2 (shuffle) - data_transform")
    print(f"  dataset={args.dataset}  horizon={args.horizon}  group={args.group}")
    print(f"  THEORY_INTERACTIONS: {len(THEORY_INTERACTIONS)} terms")
    print(f"  INNER_REPEATS: {INNER_REPEATS}")
    print("=" * 60)
    transform(args.dataset, args.horizon, args.group)


if __name__ == "__main__":
    main()
