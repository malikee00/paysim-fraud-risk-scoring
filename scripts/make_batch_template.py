from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

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
    """
    Ensure RAW_COLS exist. If nameOrig/nameDest missing -> fill blank.
    """
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
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


def _mixed_sample(df_raw: pd.DataFrame, *, n: int, seed: int, frac_focus: float) -> pd.DataFrame:
    """
    Return mixed sample:
      - focus sample: CASH_OUT/TRANSFER with higher amounts
      - random sample: uniform random
    """
    rng = np.random.default_rng(seed)
    n = min(int(n), len(df_raw))
    n_focus = int(round(n * frac_focus))
    n_rand = n - n_focus

    # ---- focus candidates ----
    focus = df_raw.copy()

    # normalize type uppercase
    focus["type"] = focus["type"].astype(str).str.upper()

    # prefer CASH_OUT/TRANSFER
    focus = focus[focus["type"].isin(["CASH_OUT", "TRANSFER"])]

    # ensure amount numeric
    focus["amount"] = pd.to_numeric(focus["amount"], errors="coerce").fillna(0.0)

    if len(focus) > 0:
        thr = focus["amount"].quantile(0.85)
        focus2 = focus[focus["amount"] >= thr]
        if len(focus2) >= max(5, n_focus // 2):
            focus = focus2

    if len(focus) >= n_focus and n_focus > 0:
        focus_sample = focus.sample(n=n_focus, random_state=seed).copy()
    else:
        focus_sample = focus.sample(n=min(n_focus, len(focus)), random_state=seed).copy()

    remaining = df_raw.drop(index=focus_sample.index, errors="ignore")
    if len(remaining) >= n_rand and n_rand > 0:
        rand_sample = remaining.sample(n=n_rand, random_state=seed + 1).copy()
    else:
        rand_sample = remaining.sample(n=min(n_rand, len(remaining)), random_state=seed + 1).copy()

    out = pd.concat([focus_sample, rand_sample], axis=0)

    if len(out) < n:
        topup = df_raw.drop(index=out.index, errors="ignore").sample(
            n=min(n - len(out), len(df_raw) - len(out)), random_state=seed + 2
        )
        out = pd.concat([out, topup], axis=0)

    out = out.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="", help="Path to parquet/csv dataset (optional)")
    parser.add_argument("--n", type=int, default=20, help="Number of demo rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--focus_frac",
        type=float,
        default=0.5,
        help="Fraction of rows from 'focus' sample (CASH_OUT/TRANSFER + high amount). Default 0.5",
    )
    args = parser.parse_args()

    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

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

    df_raw = df[RAW_COLS].copy()

    demo = _mixed_sample(df_raw, n=args.n, seed=args.seed, frac_focus=float(args.focus_frac))

    template = pd.DataFrame([{c: "" for c in RAW_COLS}])
    template.to_csv(docs_dir / "template_raw.csv", index=False)
    demo.to_csv(docs_dir / "demo_batch_raw.csv", index=False)

    print("[OK] Wrote:")
    print(" - docs/template_raw.csv")
    print(f" - docs/demo_batch_raw.csv (n={len(demo)}, focus_frac={args.focus_frac})")
    print("[INFO] Tip: upload demo_batch_raw.csv to /predict_batch?mode=raw")


if __name__ == "__main__":
    main()
