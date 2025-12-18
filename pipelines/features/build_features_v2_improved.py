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
# NEW: Enhanced Base Features (The "Sakti" Features)
# ============================================================
def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalisasi nama kolom ke lowercase dulu supaya aman
    df.columns = [c.lower() for c in df.columns]
    
    # Log amount
    df["log_amount"] = np.log1p(df["amount"])
    
    # Gunakan nama kolom huruf kecil semua (karena sudah di-lower di atas)
    # 1. Error Balance (Sakti)
    # Kita pakai .get() atau cek kolom agar tidak error
    if 'oldbalanceorg' in df.columns and 'newbalanceorig' in df.columns:
        df["error_orig"] = (df["oldbalanceorg"] - df["amount"]) - df["newbalanceorig"]
    else:
        df["error_orig"] = 0 

    if 'oldbalancedest' in df.columns and 'newbalancedest' in df.columns:
        df["error_dest"] = (df["oldbalancedest"] + df["amount"]) - df["newbalancedest"]
    else:
        df["error_dest"] = 0

    # 2. Delta Mismatch
    df["org_delta_mismatch"] = (df["delta_org"] + df["amount"]).abs()
    df["dest_delta_mismatch"] = (df["delta_dest"] - df["amount"]).abs()

    # 3. Identity Features
    # namedest juga di-lower jadi namedest
    df["is_merchant_dest"] = df["namedest"].astype(str).str.startswith("M").astype("int8")

    # 4. Binary Flags
    df["org_balance_decreased"] = (df["delta_org"] < 0).astype("int8")
    df["dest_balance_increased"] = (df["delta_dest"] > 0).astype("int8")

    # One-hot encoding untuk Type
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    df = pd.concat([df, type_dummies], axis=1)

    return df

# ============================================================
# DESTINATION + RATIO FEATURES
# ============================================================
def add_dest_velocity(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    df = df.copy()

    print("   [..] Factorizing namedest...")
    df["namedest_id"], _ = pd.factorize(df["namedest"], sort=False)
    
    print("   [..] Sorting data...")
    df.sort_values(["namedest_id", "step"], inplace=True)

    print("   [..] Shifting history...")
    df["_temp_shifted_amt"] = df.groupby("namedest_id")["amount"].shift(1)

    for w in windows:
        print(f"   [..] Processing window {w} (agg & ratios)...")
        
        # Hitung statistik dasar
        stats = (
            df.groupby("namedest_id")["_temp_shifted_amt"]
            .rolling(window=w, min_periods=1)
            .agg({"sum": "sum", "mean": "mean", "count": "count"})
            .reset_index(level=0, drop=True)
        )
        
        count_col = f"dest_txn_count_w{w}"
        mean_col = f"dest_amt_mean_w{w}"
        
        df[count_col] = stats["count"].fillna(0).astype("float32")
        df[mean_col] = stats["mean"].fillna(0).astype("float32")
        
        # 5. NEW: Ratio Features (Comparing current transaction to history)
        # Apakah transaksi ini jauh lebih besar dari rata-rata sebelumnya?
        df[f"ratio_amt_to_dest_mean_w{w}"] = (
            df["amount"] / (df[mean_col] + 1e-6)
        ).astype("float32")

    df.drop(columns=["_temp_shifted_amt", "namedest_id"], inplace=True)
    return df

def dest_unique_origin_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(["namedest", "step"], inplace=True)
    first_pair = ~df.duplicated(subset=["namedest", "nameorig"])
    cum_unique = first_pair.groupby(df["namedest"]).cumsum()
    df["dest_uniq_origs_hist"] = (
        cum_unique.groupby(df["namedest"]).shift(1).fillna(0).astype("int32")
    )
    return df

def build_feature_summary(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    summary = {"n_rows": int(len(df)), "n_features": int(len(feature_cols)), "features": {}}
    for c in feature_cols:
        s = df[c]
        summary["features"][c] = {
            "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
            "missing_rate": float(s.isna().mean()),
        }
    return summary

# ============================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Build V2 IMPROVED features.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--limit", type=int, default=0, help="Limit rows for fast test (0=full)")
    args = p.parse_args()

    print(f"[INIT] Loading data from {args.input}...")
    df = pd.read_parquet(args.input)

    # 6. NEW: Limit rows for Fast Experimenting
    if args.limit > 0:
        print(f"[FAST] Limiting to first {args.limit:,} rows...")
        df = df.head(args.limit)

    # Memory Optimization
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = df[c].astype("int32")

    target_col = "isfraud" if "isfraud" in df.columns else "isFraud"

    print("[V2] Building Enhanced Base features...")
    df_feat = build_base_features(df)

    print("[V2] Building Destination velocity + Ratios...")
    df_feat = add_dest_velocity(df_feat, windows=[10, 50]) 

    print("[V2] Building Destination unique-origin history...")
    df_feat = dest_unique_origin_history(df_feat)

    # Feature selection
    feature_cols = [
        "log_amount", "error_orig", "error_dest", "org_delta_mismatch", 
        "dest_delta_mismatch", "is_merchant_dest", "org_balance_decreased", 
        "dest_balance_increased", "dest_uniq_origs_hist"
    ] 
    # Add dynamic columns
    feature_cols += [c for c in df_feat.columns if c.startswith("type_") or "ratio_" in c or "dest_txn_count" in c]
    
    feature_cols = list(dict.fromkeys(feature_cols))
    final_cols = ["step"] + feature_cols + [target_col]
    
    df_out = df_feat[final_cols].copy()

    ensure_parent(Path(args.output))
    df_out.to_parquet(args.output, index=False)

    summary = build_feature_summary(df_out, feature_cols)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Features built. Output: {args.output}")

if __name__ == "__main__":
    main()