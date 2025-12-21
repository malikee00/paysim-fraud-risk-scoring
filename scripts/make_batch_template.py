from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_COLS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "nameOrig",
    "nameDest",
]

LABEL_COLS = ["isFraud", "isfraud"]  

CANDIDATES = [
    "data/processed_local/features_v2_full_sakti.parquet",
    "data/processed_local/features_v2_full.parquet",
    "data/processed_local/features_v2.parquet",
    "data/processed/features_v2.parquet",
    "data/processed_local/transactions_clean.parquet",
    "data/processed/transactions_clean.parquet",
    "data/raw/sample_raw.csv",
]


def _pick_existing(paths: List[str]) -> Optional[Path]:
    for p in paths:
        full = ROOT / p
        if full.exists():
            return full
    return None


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}

    rename_map = {}
    for c in RAW_COLS:
        if c in df.columns:
            continue
        if c.lower() in cols_lower:
            rename_map[cols_lower[c.lower()]] = c

    if rename_map:
        df = df.rename(columns=rename_map)

    for c in ["nameOrig", "nameDest"]:
        if c not in df.columns:
            df[c] = ""

    for lc in LABEL_COLS:
        if lc in df.columns and lc != "isFraud":
            df = df.rename(columns={lc: "isFraud"})
            break

    # verify required core columns exist
    required_core = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    missing_core = [c for c in required_core if c not in df.columns]
    if missing_core:
        raise ValueError(
            f"Dataset missing required raw columns: {missing_core}. "
            f"Columns present: {list(df.columns)[:50]}"
        )

    return df


def _sample_stratified(
    df: pd.DataFrame,
    *,
    n: int,
    seed: int,
    fraud_ratio: float,
) -> Tuple[pd.DataFrame, str]:
    """
    If isFraud exists:
      - take n_fraud = round(n*fraud_ratio) from isFraud==1
      - take n_non = n - n_fraud from isFraud==0
    If not enough fraud rows, take all fraud rows then fill remainder from non-fraud.
    """
    rng = np.random.default_rng(seed)
    n = min(int(n), len(df))

    if "isFraud" not in df.columns:
        out = df.sample(n=n, random_state=seed).reset_index(drop=True)
        return out, "random (no isFraud column found)"

    # make sure label numeric 0/1
    y = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int)
    df = df.copy()
    df["isFraud"] = y

    df_fraud = df[df["isFraud"] == 1]
    df_non = df[df["isFraud"] == 0]

    n_fraud_target = int(round(n * float(fraud_ratio)))
    n_non_target = n - n_fraud_target

    n_fraud = min(n_fraud_target, len(df_fraud))
    n_non = min(n_non_target, len(df_non))

    fraud_sample = df_fraud.sample(n=n_fraud, random_state=seed) if n_fraud > 0 else df_fraud.head(0)
    non_sample = df_non.sample(n=n_non, random_state=seed + 1) if n_non > 0 else df_non.head(0)

    out = pd.concat([fraud_sample, non_sample], axis=0)

    # top up if still short (because one class lacked rows)
    if len(out) < n:
        remaining = df.drop(index=out.index, errors="ignore")
        topup = remaining.sample(n=min(n - len(out), len(remaining)), random_state=seed + 2)
        out = pd.concat([out, topup], axis=0)

    out = out.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)

    info = f"stratified (fraud_ratio={fraud_ratio}, fraud_rows={n_fraud}, nonfraud_rows={n_non})"
    return out, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="", help="Path to parquet/csv dataset (optional)")
    parser.add_argument("--n", type=int, default=20, help="Number of demo rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--fraud_ratio",
        type=float,
        default=0.5,
        help="Desired fraud fraction in demo CSV (0..1). Default 0.5",
    )
    args = parser.parse_args()

    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # resolve dataset path
    if args.input:
        dataset_path = (ROOT / args.input).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"--input not found: {dataset_path}")
    else:
        dataset_path = _pick_existing(CANDIDATES)
        if dataset_path is None:
            raise FileNotFoundError(
                "No dataset found. Provide --input path. "
                "Tried: " + ", ".join(CANDIDATES)
            )

    print(f"[INFO] Using dataset: {dataset_path}")

    df = _read_any(dataset_path)
    df = _normalize_columns(df)

    keep_cols = RAW_COLS.copy()
    if "isFraud" in df.columns:
        keep_cols.append("isFraud")

    df_keep = df[keep_cols].copy()

    demo_df, mode_info = _sample_stratified(
        df_keep,
        n=args.n,
        seed=args.seed,
        fraud_ratio=float(args.fraud_ratio),
    )

    # write template + demo
    template = pd.DataFrame([{c: "" for c in RAW_COLS}])
    template.to_csv(docs_dir / "template_raw.csv", index=False)

    demo_raw_only = demo_df[RAW_COLS].copy()
    demo_raw_only.to_csv(docs_dir / "demo_batch_raw.csv", index=False)

    if "isFraud" in demo_df.columns:
        fraud_count = int((demo_df["isFraud"] == 1).sum())
        non_count = int((demo_df["isFraud"] == 0).sum())
        (docs_dir / "demo_batch_meta.txt").write_text(
            f"mode={mode_info}\nrows={len(demo_df)}\nfraud={fraud_count}\nnonfraud={non_count}\n",
            encoding="utf-8",
        )
        print(f"[INFO] demo label mix (from source): fraud={fraud_count}, nonfraud={non_count}")

    print("[OK] Wrote:")
    print(" - docs/template_raw.csv")
    print(f" - docs/demo_batch_raw.csv (n={len(demo_raw_only)} | {mode_info})")
    if (docs_dir / "demo_batch_meta.txt").exists():
        print(" - docs/demo_batch_meta.txt")


if __name__ == "__main__":
    main()
