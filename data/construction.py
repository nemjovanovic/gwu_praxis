#!/usr/bin/env python3
# =====================================================
# build_praxis_clean_variants.py
# Single-script builder for baseline/expanded datasets
# (without imputation). Imputation is now handled per-split
# in make_splits.py to prevent data leakage.
# =====================================================

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Paths (script-relative: this file lives in data/)
# -----------------------------
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE
DATASETS_DIR = DATA_DIR / "raw_datasets"
FULL_DIR = DATA_DIR / "constructed_datasets"
GLOBAL_DIR = DATASETS_DIR / "global"


# -----------------------------
# Helpers
# -----------------------------
def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)


def _standardize_iso3_year(
    df: pd.DataFrame,
    iso_candidates: Iterable[str] = ("ISO3", "COUNTRY.ID", "Country Code", "countrycode", "ISO", "WEOCountryCode", "ccodealp", "iso3"),
    year_candidates: Iterable[str] = ("Year", "year", "YEAR", "yr", "YR"),
) -> pd.DataFrame:
    out = df.copy()

    iso_col = next((c for c in iso_candidates if c in out.columns), None)
    if iso_col is None:
        raise KeyError("No ISO3-like column found.")
    if iso_col != "ISO3":
        out = out.rename(columns={iso_col: "ISO3"})

    year_col = next((c for c in year_candidates if c in out.columns), None)
    if year_col is None:
        raise KeyError("No Year-like column found.")
    if year_col != "Year":
        out = out.rename(columns={year_col: "Year"})

    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    return out


def _standardize_year_only(
    df: pd.DataFrame,
    year_candidates: Iterable[str] = ("Year", "year", "YEAR"),
) -> pd.DataFrame:
    out = df.copy()
    year_col = next((c for c in year_candidates if c in out.columns), None)
    if year_col is None:
        raise KeyError("No Year-like column found.")
    if year_col != "Year":
        out = out.rename(columns={year_col: "Year"})
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    return out


def _merge_on_iso_year(base: pd.DataFrame, add: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return base.merge(add[["ISO3", "Year"] + cols], on=["ISO3", "Year"], how="left")


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df



def _print_missingness(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    print(f"\nMissingness report: {label}")
    for c in cols:
        if c not in df.columns:
            print(f"  - {c}: MISSING COLUMN")
            continue
        pct = df[c].isna().mean() * 100
        print(f"  - {c}: {pct:.2f}% missing")


def _check_keys(df: pd.DataFrame, label: str) -> None:
    dupes = df.duplicated(subset=["ISO3", "Year"]).sum()
    if dupes:
        print(f"WARNING: {label}: {dupes:,} duplicate (ISO3, Year) rows found.")
    else:
        print(f"OK: {label}: no duplicate (ISO3, Year) rows.")


# -----------------------------
# Raw extraction helpers
# -----------------------------
def build_skeleton(year_start: int = 1980, year_end: int = 2021) -> pd.DataFrame:
    identity_path = DATASETS_DIR / "skeleton_identity.csv"
    identity = _read_csv(identity_path)
    identity = identity[["countryname", "ISO3", "GroupName", "wasinbankitaly"]].drop_duplicates()

    years = list(range(year_start, year_end + 1))
    expanded = (
        identity.assign(_key=1)
        .merge(pd.DataFrame({"Year": years, "_key": 1}), on="_key")
        .drop(columns=["_key"])
    )
    return expanded


def extract_weo(weo_path: Path) -> pd.DataFrame:
    indicators = [
        "GGXWDG_NGDP",  # D_WEO
        "GGXONLB_NGDP",  # PB_WEO
        "BCA_NGDPD",  # CA_WEO
        "PCPIPCH",  # CPI_WEO
        "NGDPRPPPPC",  # lnGDPpc_WEO
        "GGXONLB",  # r_g
        "GGXCNL",  # r_g
        "GGXWDG",  # r_g
    ]

    df = _read_csv(weo_path, low_memory=False)
    df = df[df["INDICATOR.ID"].isin(indicators)].copy()
    if df.empty:
        raise ValueError("No matching WEO indicators found.")

    year_cols = [c for c in df.columns if c.isdigit()]
    df_long = df.melt(
        id_vars=["COUNTRY.ID", "INDICATOR.ID"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    df_long = df_long.dropna(subset=["Value"])

    wide = (
        df_long.pivot_table(
            index=["COUNTRY.ID", "Year"],
            columns="INDICATOR.ID",
            values="Value",
        )
        .reset_index()
    )
    wide = wide.sort_values(["COUNTRY.ID", "Year"]).reset_index(drop=True)

    # lnGDPpc
    if "NGDPRPPPPC" in wide.columns:
        wide["lnGDPpc_WEO"] = np.where(
            wide["NGDPRPPPPC"] > 0,
            np.log(wide["NGDPRPPPPC"]),
            np.nan,
        )
    else:
        wide["lnGDPpc_WEO"] = np.nan

    # r_g computation
    for col in ["GGXONLB", "GGXCNL", "GGXWDG"]:
        if col not in wide.columns:
            wide[col] = np.nan

    wide["L_GGXWDG"] = wide.groupby("COUNTRY.ID")["GGXWDG"].shift(1)
    wide["interest"] = wide["GGXONLB"] - wide["GGXCNL"]
    wide["avg_debt"] = (wide["GGXWDG"] + wide["L_GGXWDG"]) / 2
    wide["r_g_WEO"] = (wide["interest"] / wide["avg_debt"]) * 100
    wide["r_g_WEO"] = wide["r_g_WEO"].replace([np.inf, -np.inf], np.nan)

    # Country median fill (as in original)
    med = wide.groupby("COUNTRY.ID")["r_g_WEO"].median()
    wide["r_g_WEO"] = wide.apply(
        lambda row: med[row["COUNTRY.ID"]] if pd.isna(row["r_g_WEO"]) else row["r_g_WEO"],
        axis=1,
    )

    out = wide.rename(columns={
        "COUNTRY.ID": "ISO3",
        "GGXWDG_NGDP": "D_WEO",
        "GGXONLB_NGDP": "PB_WEO",
        "BCA_NGDPD": "CA_WEO",
        "PCPIPCH": "CPI_WEO",
    })

    cols = ["ISO3", "Year", "D_WEO", "PB_WEO", "CA_WEO", "CPI_WEO", "lnGDPpc_WEO", "r_g_WEO"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def extract_wdi(wdi_path: Path) -> pd.DataFrame:
    codes = [
        "DT.DOD.DECT.CD",  # external debt current USD
        "NY.GDP.MKTP.CD",  # GDP current USD
        "DT.TDS.DPPF.XP.ZS",  # debt service
        "BX.TRF.PWKR.DT.GD.ZS",  # remittances
        "FI.RES.TOTL.MO",  # FX reserves
    ]
    df = _read_csv(wdi_path)
    df = df[df["Indicator Code"].isin(codes)].copy()
    if df.empty:
        raise ValueError("No matching WDI indicators found.")

    year_cols = [c for c in df.columns if c.isdigit()]
    df_long = df.melt(
        id_vars=["Country Code", "Indicator Code"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    df_long = df_long.dropna(subset=["Value"])

    wide = (
        df_long.pivot_table(
            index=["Country Code", "Year"],
            columns="Indicator Code",
            values="Value",
        )
        .reset_index()
    )

    # Compute ED_WB = external debt / GDP * 100
    wide["ED_WB"] = (wide.get("DT.DOD.DECT.CD") / wide.get("NY.GDP.MKTP.CD")) * 100
    wide = wide.rename(columns={
        "Country Code": "ISO3",
        "DT.TDS.DPPF.XP.ZS": "DSED_WB",
        "BX.TRF.PWKR.DT.GD.ZS": "Rem_WB",
        "FI.RES.TOTL.MO": "FXR_WB",
    })

    cols = ["ISO3", "Year", "ED_WB", "DSED_WB", "Rem_WB", "FXR_WB"]
    for c in cols:
        if c not in wide.columns:
            wide[c] = np.nan
    return wide[cols]


def extract_gmd(gmd_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "countryname", "ISO3", "year",
        "gen_govdebt_GDP", "infl", "CA_GDP", "REER",
        "deflator", "cons_GDP", "inv_GDP", "finv_GDP", "exports_GDP", "imports_GDP",
        "govtax_GDP", "M0", "M1", "M2", "cbrate", "strate", "unemp", "nGDP",
    ]
    gmd = _read_csv(gmd_path, usecols=cols)
    gmd = gmd.rename(columns={"countryname": "Country", "ISO3": "ISO3", "year": "Year"})
    gmd["Year"] = pd.to_numeric(gmd["Year"], errors="coerce")

    # Base series
    base = gmd[["ISO3", "Year", "gen_govdebt_GDP", "infl", "CA_GDP", "REER"]].copy()
    base = base.sort_values(["ISO3", "Year"]).reset_index(drop=True)

    base["D_GMD"] = base["gen_govdebt_GDP"]
    base["CPI_GMD"] = base["infl"]
    base["CA_GMD"] = base["CA_GDP"]
    base["DlnREER_GMD"] = base.groupby("ISO3")["REER"].transform(
        lambda x: np.log(x).diff()
    )

    base_out = base[["ISO3", "Year", "D_GMD", "CPI_GMD", "CA_GMD", "DlnREER_GMD"]]

    # Expanded series
    exp = gmd[[
        "ISO3", "Year",
        "deflator", "cons_GDP", "inv_GDP", "finv_GDP", "exports_GDP", "imports_GDP",
        "govtax_GDP", "M0", "M1", "M2", "cbrate", "strate", "unemp", "nGDP",
    ]].copy()

    for m in ["M0", "M1", "M2"]:
        exp[f"{m}_GDP"] = (exp[m] / exp["nGDP"]) * 100

    raw_vars = [
        "deflator", "cons_GDP", "inv_GDP", "finv_GDP", "exports_GDP", "imports_GDP",
        "govtax_GDP", "M0", "M1", "M2", "M0_GDP", "M1_GDP", "M2_GDP",
        "cbrate", "strate", "unemp", "nGDP",
    ]
    rename_map = {v: f"{v}_GMD" for v in raw_vars}
    exp = exp.rename(columns=rename_map)

    return base_out, exp


def extract_wgi(wgi_path: Path) -> pd.DataFrame:
    df = _read_csv(wgi_path, usecols=["countryname", "code", "year", "estimate"])
    df = df.rename(columns={"code": "ISO3", "year": "Year", "estimate": "gee"})
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df[["ISO3", "Year", "gee"]]


def extract_kaopen(ka_path: Path) -> pd.DataFrame:
    df = _read_csv(ka_path, usecols=["country_name", "ccode", "year", "ka_open"])
    df = df.rename(columns={"ccode": "ISO3", "year": "Year"})
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df[["ISO3", "Year", "ka_open"]]


def extract_qog(qog_path: Path) -> pd.DataFrame:
    qog = _read_csv(qog_path, low_memory=False)
    iso_candidates = ["ccodealp", "CCODEALP", "iso3", "ISO3"]
    year_candidates = ["year", "Year", "yr", "YR"]

    iso_col = next((c for c in iso_candidates if c in qog.columns), None)
    year_col = next((c for c in year_candidates if c in qog.columns), None)
    if iso_col is None or year_col is None:
        raise KeyError("Could not find ISO3/Year columns in QoG.")

    vars_needed = ["fh_pr", "van_index", "cbie_lvau", "cbie_lvaw", "wdi_birthskill"]
    available = [v for v in vars_needed if v in qog.columns]

    qog = qog[[iso_col, year_col] + available].copy()
    qog = qog.rename(columns={iso_col: "ISO3", year_col: "Year"})
    qog["Year"] = pd.to_numeric(qog["Year"], errors="coerce")
    qog = qog.dropna(subset=["ISO3", "Year"]).copy()

    rename_map = {v: f"{v}_QOG" for v in available}
    qog = qog.rename(columns=rename_map)

    return qog


def extract_crisis(crisis_path: Path) -> pd.DataFrame:
    crisis = _read_csv(crisis_path)
    crisis = _standardize_iso3_year(crisis, iso_candidates=("ISO", "WEOCountryCode", "ISO3"), year_candidates=("year", "Year"))
    crisis = _coerce_numeric(crisis, ["c1", "c2"])
    crisis["crisis"] = ((crisis["c1"] == 1) | (crisis["c2"] == 1)).astype(int)
    return crisis[["ISO3", "Year", "c1", "c2", "crisis"]]


def extract_esdb(esdb_path: Path) -> pd.DataFrame:
    esdb = _read_csv(esdb_path)
    esdb = _standardize_iso3_year(esdb, iso_candidates=("ISO3",), year_candidates=("year", "Year"))

    cols = [
        "ISO3", "Year",
        "dep_totl", "dep_old", "clm_govt", "clm_priv",
        "net_oda", "free_ovr", "free_civ", "crd_priv",
    ]
    for c in cols:
        if c not in esdb.columns:
            esdb[c] = pd.NA
    esdb = esdb[cols].copy()

    return esdb


def extract_global_series(global_dir: Path) -> pd.DataFrame:
    def _try_read(path: Path, label: str) -> pd.DataFrame | None:
        if not path.exists():
            print(f"WARNING: Missing global file: {path} (column {label} will be NA)")
            return None
        return _standardize_year_only(_read_csv(path))

    lnvix = _try_read(global_dir / "lnVIX.csv", "lnVIX")
    wgdp = _try_read(global_dir / "WGDPp.csv", "WGDPp")
    rshort = _try_read(global_dir / "short_rate_annual.csv", "Rshort")
    rlong = _try_read(global_dir / "long_rate_annual.csv", "Rlong")
    poil = _try_read(global_dir / "DlnPoil.csv", "DlnPoil")
    pcom = _try_read(global_dir / "DlnPcom.csv", "DlnPcom")

    frames = []
    if lnvix is not None:
        frames.append(lnvix.rename(columns={"lnVIX": "lnVIX"})[["Year", "lnVIX"]])
    if wgdp is not None:
        frames.append(wgdp.rename(columns={"WGDPg": "WGDPg"})[["Year", "WGDPg"]])
    if rshort is not None:
        frames.append(rshort.rename(columns={"ShortRate_annual": "Rshort"})[["Year", "Rshort"]])
    if rlong is not None:
        frames.append(rlong.rename(columns={"LongRate_annual": "Rlong"})[["Year", "Rlong"]])
    if poil is not None:
        frames.append(poil.rename(columns={"DlnPoil": "DlnPoil"})[["Year", "DlnPoil"]])
    if pcom is not None:
        frames.append(pcom.rename(columns={"DlnPcom": "DlnPcom"})[["Year", "DlnPcom"]])

    if not frames:
        raise FileNotFoundError(f"No global series files found in: {global_dir}")

    out = frames[0].copy()
    for f in frames[1:]:
        out = out.merge(f, on="Year", how="outer")

    # Ensure all expected columns exist
    for col in ["lnVIX", "WGDPg", "Rshort", "Rlong", "DlnPoil", "DlnPcom"]:
        if col not in out.columns:
            out[col] = np.nan

    return out


# -----------------------------
# Build Baseline (no imputation)
# -----------------------------
def build_baseline_without_imputation(
    skeleton: pd.DataFrame,
    weo: pd.DataFrame,
    wdi: pd.DataFrame,
    gmd_base: pd.DataFrame,
    wgi: pd.DataFrame,
    kaopen: pd.DataFrame,
    crisis: pd.DataFrame,
    global_series: pd.DataFrame,
) -> pd.DataFrame:
    print("Building baseline_without_imputation.csv...")

    base = skeleton[["ISO3", "GroupName", "Year"]].copy()

    base = _merge_on_iso_year(base, weo, ["D_WEO", "PB_WEO", "CA_WEO", "CPI_WEO", "lnGDPpc_WEO", "r_g_WEO"])
    base = _merge_on_iso_year(base, wdi, ["ED_WB", "DSED_WB", "Rem_WB", "FXR_WB"])
    base = _merge_on_iso_year(base, gmd_base, ["D_GMD", "CPI_GMD", "CA_GMD", "DlnREER_GMD"])
    base = _merge_on_iso_year(base, wgi, ["gee"])
    base = _merge_on_iso_year(base, kaopen, ["ka_open"])

    # Construct merged/raw final columns (no imputation)
    base["D_WEO"] = pd.to_numeric(base["D_WEO"], errors="coerce")
    base["D_GMD"] = pd.to_numeric(base["D_GMD"], errors="coerce")
    base["D"] = base["D_WEO"].fillna(base["D_GMD"])
    base["ED"] = base["ED_WB"]
    base["DSED"] = base["DSED_WB"]
    base["PB"] = base["PB_WEO"]
    base["r_g"] = base["r_g_WEO"]
    base["Rem"] = base["Rem_WB"]
    base["FXR"] = base["FXR_WB"]
    base["lnGDPpc"] = base["lnGDPpc_WEO"]
    base["CPI"] = base["CPI_WEO"].fillna(base["CPI_GMD"])
    base["DlnREER"] = base["DlnREER_GMD"]
    base["CA"] = base["CA_WEO"].fillna(base["CA_GMD"])

    # Global series
    base = base.merge(global_series, on="Year", how="left")

    # Crisis variables
    base = base.merge(crisis[["ISO3", "Year", "crisis"]], on=["ISO3", "Year"], how="left")
    base["crisis"] = base["crisis"].fillna(0).astype(int)

    # CH (Crisis History)
    base = base.sort_values(["ISO3", "Year"]).reset_index(drop=True)
    base["CH"] = (
        base.groupby("ISO3")["crisis"]
        .transform(lambda x: x.rolling(window=10, min_periods=1).max().shift(1))
        .fillna(0)
        .astype(int)
    )

    # TC (Share of Countries in Crisis)
    tc = (
        base.groupby("Year")
        .agg(total_crisis=("crisis", "sum"), total_countries=("ISO3", "nunique"))
        .reset_index()
    )
    tc["TC"] = tc["total_crisis"] / tc["total_countries"]
    base = base.merge(tc[["Year", "TC"]], on="Year", how="left")

    final_cols = [
        "ISO3", "GroupName", "Year",
        "D", "ED", "DSED", "PB", "r_g", "Rem", "FXR", "lnGDPpc",
        "CPI", "DlnREER", "CA", "ka_open", "gee",
        "lnVIX", "WGDPg", "Rshort", "Rlong", "DlnPoil",
        "crisis", "CH", "TC", "DlnPcom",
    ]
    for c in final_cols:
        if c not in base.columns:
            base[c] = pd.NA

    base = base[final_cols].copy()
    base = base.sort_values(["GroupName", "ISO3", "Year"]).reset_index(drop=True)

    out_path = FULL_DIR / "baseline_without_imputation.csv"
    base.rename(columns={"ISO3": "WEOCountryCode"}).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return base



# -----------------------------
# Expanded Without Imputation
# -----------------------------
def build_expanded_without_imputation(
    skeleton: pd.DataFrame,
    gmd_expanded: pd.DataFrame,
    qog: pd.DataFrame,
    baseline_no_imp: pd.DataFrame,
    esdb: pd.DataFrame,
) -> pd.DataFrame:
    print("Building expanded_without_imputation.csv...")

    merged = skeleton.merge(gmd_expanded, on=["ISO3", "Year"], how="left")
    merged = merged.merge(qog, on=["ISO3", "Year"], how="left")

    gmd_cols = [c for c in merged.columns if c.endswith("_GMD")]
    for src in gmd_cols:
        merged[src.replace("_GMD", "")] = merged[src]

    qog_cols = [c for c in merged.columns if c.endswith("_QOG")]
    for src in qog_cols:
        merged[src.replace("_QOG", "")] = merged[src]

    base = baseline_no_imp.copy()
    base = base.sort_values(["ISO3", "Year"]).reset_index(drop=True)
    merged = merged.merge(base, on=["ISO3", "Year"], how="left", suffixes=("", "_base"))

    esdb = esdb.sort_values(["ISO3", "Year"]).reset_index(drop=True)
    merged = merged.merge(esdb, on=["ISO3", "Year"], how="left", suffixes=("", "_esdb"))

    final_cols = [
        "ISO3", "GroupName", "Year",
        "D", "ED", "DSED", "PB", "r_g", "Rem", "FXR", "lnGDPpc",
        "CPI", "DlnREER", "CA", "ka_open", "gee",
        "lnVIX", "WGDPg", "Rshort", "Rlong", "DlnPoil",
        "crisis", "CH", "TC", "DlnPcom",
        "deflator", "cons_GDP", "finv_GDP", "exports_GDP", "imports_GDP",
        "govtax_GDP", "M2_GDP", "cbrate", "strate", "unemp",
        "fh_pr", "van_index", "cbie_lvau", "wdi_birthskill",
        "dep_totl", "dep_old", "clm_govt", "clm_priv",
        "net_oda", "free_ovr", "free_civ", "crd_priv",
    ]
    for c in final_cols:
        if c not in merged.columns:
            merged[c] = pd.NA

    merged = merged[final_cols].copy()
    merged = merged.sort_values(["GroupName", "ISO3", "Year"]).reset_index(drop=True)

    out_path = FULL_DIR / "expanded_without_imputation.csv"
    merged.rename(columns={"ISO3": "WEOCountryCode"}).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return merged



# -----------------------------
# Clean Outputs
# -----------------------------
# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("Starting PRAXIS clean variants build (raw-data driven).")
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    skeleton = build_skeleton(1980, 2021)
    weo = extract_weo(DATASETS_DIR / "WEO_CSV_OCT_25" / "WEO_OCT_2025.csv")
    wdi = extract_wdi(DATASETS_DIR / "WDI_CSV_10_08" / "WDICSV.csv")
    gmd_base, gmd_expanded = extract_gmd(DATASETS_DIR / "GMD" / "GMD.csv")
    wgi = extract_wgi(DATASETS_DIR / "WGI" / "WGI_ge.csv")
    kaopen = extract_kaopen(DATASETS_DIR / "ka_open" / "ka_open.csv")
    qog = extract_qog(DATASETS_DIR / "QoG" / "qog_std_ts_jan25.csv")
    crisis = extract_crisis(DATASETS_DIR / "MoroDeMarchi_Crisis" / "crisis_morodemarchi.csv")
    global_series = extract_global_series(GLOBAL_DIR)
    esdb = extract_esdb(DATASETS_DIR / "esdb" / "super_selection_database_wide.csv")

    baseline_no_imp = build_baseline_without_imputation(
        skeleton, weo, wdi, gmd_base, wgi, kaopen, crisis, global_series
    )
    expanded_no_imp = build_expanded_without_imputation(skeleton, gmd_expanded, qog, baseline_no_imp, esdb)

    print("\nValidation summary:")
    print(f"Skeleton rows: {len(skeleton):,}")
    print(f"Baseline rows: {len(baseline_no_imp):,}")
    print(f"Expanded rows: {len(expanded_no_imp):,}")

    _check_keys(baseline_no_imp, "Baseline")
    _check_keys(expanded_no_imp, "Expanded")

    _print_missingness(
        baseline_no_imp,
        ["D", "ED", "DSED", "PB", "r_g", "Rem", "FXR", "lnGDPpc", "CPI", "DlnREER", "CA", "ka_open", "gee"],
        "Baseline (no imputation)"
    )
    _print_missingness(
        expanded_no_imp,
        [
            "D", "ED", "DSED", "PB", "r_g", "Rem", "FXR", "lnGDPpc",
            "CPI", "DlnREER", "CA", "ka_open", "gee",
            "deflator", "cons_GDP", "finv_GDP", "exports_GDP", "imports_GDP",
            "govtax_GDP", "M2_GDP", "cbrate", "strate", "unemp",
            "fh_pr", "van_index", "cbie_lvau", "wdi_birthskill",
            "dep_totl", "dep_old", "clm_govt", "clm_priv",
            "net_oda", "free_ovr", "free_civ", "crd_priv",
        ],
        "Expanded (no imputation)"
    )

    print("\nAll outputs built successfully.")


if __name__ == "__main__":
    main()
