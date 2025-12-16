from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


REQUIRED_COLS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]

OPTIONAL_COLS = ["isFlaggedFraud"]


@dataclass
class ValidationConfig:
    max_missing_rate_key_cols: float = 0.01  
    max_missing_rate_any_col: float = 0.05   
    allow_amount_zero: bool = True
    allowed_fraud_values: Tuple[int, int] = (0, 1)

def missing_rates(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    rates = {}
    n = len(df)
    for c in cols:
        if c not in df.columns:
            continue
        rates[c] = float(df[c].isna().sum()) / float(n) if n > 0 else 0.0
    return rates


def fail(errors: List[str], msg: str) -> None:
    errors.append(msg)


def validate_schema(
    df: pd.DataFrame,
    config: ValidationConfig,
    strict: bool = True,
) -> Tuple[bool, List[str], Dict[str, float]]:
    errors: List[str] = []
    warnings: List[str] = []

    # 1) Required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        fail(errors, f"Missing required columns: {missing}")
        return False, errors, {}

    # 2) Type checks (lightweight, not overly strict)
    if df["step"].isna().any():
        fail(errors, "Column 'step' contains missing values.")
    if not pd.api.types.is_numeric_dtype(df["amount"]):
        fail(errors, "Column 'amount' must be numeric.")
    else:
        if (df["amount"] < 0).any():
            fail(errors, "Column 'amount' has negative values (must be >= 0).")

    balance_cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    for c in balance_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            fail(errors, f"Column '{c}' must be numeric.")
        if df[c].isna().any():
            fail(errors, f"Column '{c}' contains missing values (balance columns must not be NaN).")

    if df["type"].isna().any():
        fail(errors, "Column 'type' contains missing values.")
    if df["isFraud"].isna().any():
        fail(errors, "Column 'isFraud' contains missing values.")
    else:
        allowed = set(config.allowed_fraud_values)
        bad = df.loc[~df["isFraud"].isin(allowed), "isFraud"].unique().tolist()
        if bad:
            fail(errors, f"Column 'isFraud' contains values outside {sorted(list(allowed))}: {bad}")

    # 3) Missing rate checks
    key_cols = ["step", "type", "amount", "isFraud"] + balance_cols
    key_rates = missing_rates(df, key_cols)
    for c, r in key_rates.items():
        if r > config.max_missing_rate_key_cols:
            fail(errors, f"Missing rate too high for key col '{c}': {r:.2%} > {config.max_missing_rate_key_cols:.2%}")

    req_rates = missing_rates(df, REQUIRED_COLS)
    for c, r in req_rates.items():
        if r > config.max_missing_rate_any_col:
            fail(errors, f"Missing rate too high for required col '{c}': {r:.2%} > {config.max_missing_rate_any_col:.2%}")

    # 4) Simple value sanity checks
    if pd.api.types.is_numeric_dtype(df["amount"]) and len(df) > 0:
        p99 = float(df["amount"].quantile(0.99))
        if p99 > 1e7:
            warnings.append(f"Warning: 99th percentile of amount is very high: {p99:.2f} (check units/outliers).")

    for oc in OPTIONAL_COLS:
        if oc not in df.columns:
            warnings.append(f"Warning: optional column '{oc}' not found (OK).")

    if warnings:
        errors.extend(warnings)

    ok = len([e for e in errors if not e.startswith("Warning:")]) == 0
    if strict and not ok:
        return False, errors, req_rates
    return True, errors, req_rates


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate PaySim CSV schema and basic data quality.")
    p.add_argument("--input", required=True, help="Path to raw CSV (sample or full).")
    p.add_argument("--max_missing_key", type=float, default=0.01, help="Max missing rate for key cols (default 0.01).")
    p.add_argument("--max_missing_any", type=float, default=0.05, help="Max missing rate for any required col (default 0.05).")
    p.add_argument("--strict", action="store_true", help="Fail (exit 1) if validation errors found.")
    p.add_argument("--nrows", type=int, default=None, help="Optional: read only first N rows (faster for huge CSV).")
    return p

def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, nrows=args.nrows)

    config = ValidationConfig(
        max_missing_rate_key_cols=args.max_missing_key,
        max_missing_rate_any_col=args.max_missing_any,
    )

    ok, messages, rates = validate_schema(df, config=config, strict=args.strict)

    print("=== Schema Validation Report ===")
    print(f"File: {input_path}")
    print(f"Rows read: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    if rates:
        print("\nMissing rates (required cols):")
        for c, r in sorted(rates.items(), key=lambda x: x[0]):
            print(f"- {c:15s}: {r:.2%}")

    if messages:
        print("\nFindings:")
        for m in messages:
            print(f"- {m}")

    if ok:
        print("\nPASS: schema/data checks look good.")
        raise SystemExit(0)
    else:
        print("\nFAIL: schema/data checks failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
