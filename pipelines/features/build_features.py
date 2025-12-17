from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---------- Numerical ----------
    df["log_amount"] = np.log1p(df["amount"])

    # delta_org, delta_dest already exist from transform
    # keep them as-is

    # mismatch features
    # org: money leaves origin -> newbalance = oldbalance - amount
    df["org_delta_mismatch"] = (df["delta_org"] + df["amount"]).abs()

    # dest: money arrives at destination -> newbalance = oldbalance + amount
    df["dest_delta_mismatch"] = (df["delta_dest"] - df["amount"]).abs()

    # ---------- Flags ----------
    df["org_balance_decreased"] = (df["delta_org"] < 0).astype("int8")
    df["dest_balance_increased"] = (df["delta_dest"] > 0).astype("int8")

    # ---------- One-hot encoding for type ----------
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    df = pd.concat([df, type_dummies], axis=1)

    return df


def build_feature_summary(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    summary = {
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "features": {},
    }

    for c in feature_cols:
        if c not in df.columns:
            continue
        series = df[c]
        if pd.api.types.is_numeric_dtype(series):
            summary["features"][c] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "missing_rate": float(series.isna().mean()),
            }
        else:
            summary["features"][c] = {
                "unique_values": int(series.nunique()),
                "missing_rate": float(series.isna().mean()),
            }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline (V1) features from canonical dataset.")
    parser.add_argument("--input", required=True, help="Input parquet (transactions_clean.parquet)")
    parser.add_argument("--output", required=True, help="Output features parquet")
    parser.add_argument("--summary", required=True, help="Output feature summary JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)

    df_feat = build_features(df)

    # Define final feature columns (explicit & ordered)
    feature_cols = [
        "log_amount",
        "delta_org",
        "delta_dest",
        "org_delta_mismatch",
        "dest_delta_mismatch",
        "org_balance_decreased",
        "dest_balance_increased",
    ] + [c for c in df_feat.columns if c.startswith("type_")]

    META_COLS = ["step"]
    TARGET_COL = "isFraud" 

    if TARGET_COL not in df_feat.columns and "isfraud" in df_feat.columns:
        TARGET_COL = "isfraud"

    final_cols = META_COLS + feature_cols + [TARGET_COL]

    missing = [c for c in final_cols if c not in df_feat.columns]
    if missing:
        raise KeyError(f"Missing columns for output: {missing}")

    df_out = df_feat[final_cols]


    ensure_parent(output_path)
    df_out.to_parquet(output_path, index=False)

    summary = build_feature_summary(df_out, feature_cols)
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(" Feature building V1 completed")
    print(f"- Input   : {input_path}")
    print(f"- Output  : {output_path}")
    print(f"- Rows    : {len(df_out)}")
    print(f"- Features: {len(feature_cols)}")


if __name__ == "__main__":
    main()
