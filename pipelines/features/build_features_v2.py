from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Base (event-level) features
# ============================================================
def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_amount"] = np.log1p(df["amount"])
    df["org_delta_mismatch"] = (df["delta_org"] + df["amount"]).abs()
    df["dest_delta_mismatch"] = (df["delta_dest"] - df["amount"]).abs()

    df["org_balance_decreased"] = (df["delta_org"] < 0).astype("int8")
    df["dest_balance_increased"] = (df["delta_dest"] > 0).astype("int8")

    type_dummies = pd.get_dummies(df["type"], prefix="type")
    df = pd.concat([df, type_dummies], axis=1)

    return df


# ============================================================
# DESTINATION 
# ============================================================
def add_dest_velocity(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Destination-only behavioral features.
    History-only (shift(1)) => no leakage.
    Optimized with .agg() for single-pass calculation.
    """
    df = df.copy()

    print("   [..] Factorizing namedest...")
    df["namedest_id"], _ = pd.factorize(df["namedest"], sort=False)
    
    if df["namedest_id"].max() < 2147483647:
        df["namedest_id"] = df["namedest_id"].astype("int32")

    print("   [..] Sorting data...")
    df.sort_values(["namedest_id", "step"], inplace=True)

    print("   [..] Shifting history...")
    df["_temp_shifted_amt"] = df.groupby("namedest_id")["amount"].shift(1)

    for w in windows:
        print(f"   [..] Processing window {w} (agg)...")
        
        stats = (
            df.groupby("namedest_id")["_temp_shifted_amt"]
            .rolling(window=w, min_periods=1)
            .agg({
                "sum": "sum",
                "mean": "mean",
                "count": "count"
            })
            .reset_index(level=0, drop=True)
        )
        
        df[f"dest_txn_count_w{w}"] = stats["count"].fillna(0).astype("float32")
        df[f"dest_amt_sum_w{w}"]   = stats["sum"].fillna(0).astype("float32")
        df[f"dest_amt_mean_w{w}"]  = stats["mean"].fillna(0).astype("float32")
        
        df[f"dest_is_burst_w{w}"] = (df[f"dest_txn_count_w{w}"] >= 5).astype("int8")

    df.drop(columns=["_temp_shifted_amt", "namedest_id"], inplace=True)

    return df


# ============================================================
# FAST unique origin history per destination
# ============================================================
def dest_unique_origin_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(["namedest", "step"], inplace=True)

    first_pair = ~df.duplicated(subset=["namedest", "nameorig"])
    cum_unique = first_pair.groupby(df["namedest"]).cumsum()

    df["dest_uniq_origs_hist"] = (
        cum_unique.groupby(df["namedest"]).shift(1).fillna(0).astype("int32")
    )

    return df


# ============================================================
def build_feature_summary(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    summary = {
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "features": {},
    }
    for c in feature_cols:
        s = df[c]
        summary["features"][c] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "missing_rate": float(s.isna().mean()),
        }
    return summary


# ============================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="Build V2 DEST-ONLY behavioral features (PaySim-optimized, fast)."
    )
    p.add_argument("--input", required=True, help="transactions_clean_full.parquet")
    p.add_argument("--output", required=True, help="features_v2_full.parquet")
    p.add_argument("--summary", required=True, help="feature_summary_v2_full.json")
    args = p.parse_args()

    df = pd.read_parquet(args.input)

    print("[INIT] Optimizing memory usage...")
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = df[c].astype("int32")

    required = ["step", "amount", "type", "nameorig", "namedest", "delta_org", "delta_dest"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    target_col = "isfraud" if "isfraud" in df.columns else "isFraud"

    print("[V2 DEST] Base features...")
    df_feat = build_base_features(df)

    print("[V2 DEST] Destination velocity...")
    df_feat = add_dest_velocity(df_feat, windows=[10, 50, 200])

    print("[V2 DEST] Destination unique-origin history...")
    df_feat = dest_unique_origin_history(df_feat)

    # Feature selection
    base_cols = [
        "log_amount",
        "delta_org",
        "delta_dest",
        "org_delta_mismatch",
        "dest_delta_mismatch",
        "org_balance_decreased",
        "dest_balance_increased",
    ] + [c for c in df_feat.columns if c.startswith("type_")]

    dest_behavioral_cols = [
        c for c in df_feat.columns
        if c.startswith("dest_")
        and c not in {"dest_delta_mismatch", "dest_balance_increased"}
    ]

    feature_cols = base_cols + dest_behavioral_cols
    feature_cols = list(dict.fromkeys(feature_cols))  

    final_cols = ["step"] + feature_cols + [target_col]
    df_out = df_feat[final_cols].copy()

    out_path = Path(args.output)
    sum_path = Path(args.summary)

    ensure_parent(out_path)
    df_out.to_parquet(out_path, index=False)

    summary = build_feature_summary(df_out, feature_cols)
    ensure_parent(sum_path)
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[OK] V2 DEST-ONLY features built")
    print(f"- Rows     : {len(df_out):,}")
    print(f"- Features : {len(feature_cols)}")
    print(f"- Output   : {out_path}")


if __name__ == "__main__":
    main()
