from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd

@dataclass
class TransformConfig:
    drop_optional_cols: bool = False

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Step: int
    if "step" in df.columns:
        df["step"] = pd.to_numeric(df["step"], errors="raise").astype("int64")

    # Type: string 
    if "type" in df.columns:
        df["type"] = df["type"].astype("string")

    # Amount/balances: float
    float_cols = ["amount", "oldbalanceorg", "newbalanceorig", "oldbalancedest", "newbalancedest"]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")

    # Labels: int
    if "isfraud" in df.columns:
        df["isfraud"] = pd.to_numeric(df["isfraud"], errors="raise").astype("int64")

    # Optional column
    if "isflaggedfraud" in df.columns:
        df["isflaggedfraud"] = pd.to_numeric(df["isflaggedfraud"], errors="coerce").fillna(0).astype("int64")

    return df


def add_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper columns (not advanced ML features yet):
    - delta_org = newbalanceorig - oldbalanceorg
    - delta_dest = newbalancedest - oldbalancedest
    """
    df = df.copy()

    required_org = {"newbalanceorig", "oldbalanceorg"}
    required_dest = {"newbalancedest", "oldbalancedest"}

    if required_org.issubset(set(df.columns)):
        df["delta_org"] = df["newbalanceorig"] - df["oldbalanceorg"]
    else:
        df["delta_org"] = pd.NA

    if required_dest.issubset(set(df.columns)):
        df["delta_dest"] = df["newbalancedest"] - df["oldbalancedest"]
    else:
        df["delta_dest"] = pd.NA

    return df


def drop_columns(df: pd.DataFrame, config: TransformConfig) -> pd.DataFrame:
    df = df.copy()
    if config.drop_optional_cols:
        for c in ["isflaggedfraud"]:
            if c in df.columns:
                df = df.drop(columns=[c])
    return df

def transform(
    input_csv: Path,
    output_parquet: Path,
    config: TransformConfig,
    nrows: Optional[int] = None,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, nrows=nrows)
    df = standardize_columns(df)
    df = cast_types(df)
    df = add_helper_columns(df)
    df = drop_columns(df, config)

    ensure_parent(output_parquet)
    df.to_parquet(output_parquet, index=False)

    print(" Transform complete")
    print(f"- Input  : {input_csv}")
    print(f"- Output : {output_parquet}")
    print(f"- Rows   : {len(df)}")
    print(f"- Cols   : {len(df.columns)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Transform PaySim raw CSV into clean canonical parquet.")
    p.add_argument("--input", required=True, help="Input raw CSV path (sample or full).")
    p.add_argument("--output", required=True, help="Output parquet path")
    p.add_argument("--drop_optional", action="store_true", help="Drop optional cols.")
    p.add_argument("--nrows", type=int, default=None, help="Optional: read only first N rows (faster for huge CSV).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    config = TransformConfig(drop_optional_cols=args.drop_optional)
    transform(
        input_csv=Path(args.input),
        output_parquet=Path(args.output),
        config=config,
        nrows=args.nrows,
    )


if __name__ == "__main__":
    main()
