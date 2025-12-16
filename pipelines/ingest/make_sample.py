from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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

@dataclass
class SampleMeta:
    created_at_utc: str
    input_path: str
    output_path: str
    seed: int
    n_fraud: int
    n_nonfraud: int
    total_rows: int
    fraud_rate: float
    step_min: int
    step_max: int
    columns: list[str]

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

    if df["isFraud"].dropna().isin([0, 1]).mean() < 1.0:
        bad = df.loc[~df["isFraud"].isin([0, 1]), "isFraud"].unique().tolist()
        raise ValueError(f"isFraud contains non-binary values: {bad}")

    if df["step"].isna().any():
        raise ValueError("step has missing values.")

    if (df["amount"] < 0).any():
        raise ValueError("amount has negative values.")

def filter_step_window(df: pd.DataFrame, step_min: Optional[int], step_max: Optional[int]) -> pd.DataFrame:
    if step_min is None and step_max is None:
        return df

    smin = int(df["step"].min()) if step_min is None else step_min
    smax = int(df["step"].max()) if step_max is None else step_max
    if smin > smax:
        raise ValueError(f"step_min ({smin}) cannot be > step_max ({smax})")

    out = df[(df["step"] >= smin) & (df["step"] <= smax)].copy()
    if out.empty:
        raise ValueError(f"No rows after step filter [{smin}, {smax}]")
    return out

def make_sample(
    input_csv: Path,
    output_csv: Path,
    seed: int,
    n_fraud: int,
    n_nonfraud: int,
    meta_json: Optional[Path],
    step_min: Optional[int],
    step_max: Optional[int],
) -> SampleMeta:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    validate_schema(df)
    df = filter_step_window(df, step_min, step_max)

    fraud = df[df["isFraud"] == 1]
    nonfraud = df[df["isFraud"] == 0]

    if len(fraud) < n_fraud:
        raise ValueError(f"Not enough fraud rows: requested {n_fraud}, available {len(fraud)}")
    if len(nonfraud) < n_nonfraud:
        raise ValueError(f"Not enough non-fraud rows: requested {n_nonfraud}, available {len(nonfraud)}")

    sample_fraud = fraud.sample(n=n_fraud, random_state=seed)
    sample_nonfraud = nonfraud.sample(n=n_nonfraud, random_state=seed)

    sample = pd.concat([sample_fraud, sample_nonfraud], axis=0)
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    ensure_parent(output_csv)
    sample.to_csv(output_csv, index=False)

    meta = SampleMeta(
        created_at_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        input_path=str(input_csv),
        output_path=str(output_csv),
        seed=seed,
        n_fraud=n_fraud,
        n_nonfraud=n_nonfraud,
        total_rows=int(len(sample)),
        fraud_rate=float(sample["isFraud"].mean()),
        step_min=int(sample["step"].min()),
        step_max=int(sample["step"].max()),
        columns=list(sample.columns),
    )

    if meta_json is not None:
        ensure_parent(meta_json)
        meta_json.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    return meta

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create a reproducible PaySim sample CSV.")
    p.add_argument("--input", required=True, help="Full PaySim CSV path (local only)")
    p.add_argument("--output", required=True, help="Sample output CSV path")
    p.add_argument("--meta", default=None, help="Optional metadata JSON output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_fraud", type=int, default=500)
    p.add_argument("--n_nonfraud", type=int, default=9500)
    p.add_argument("--step_min", type=int, default=None)
    p.add_argument("--step_max", type=int, default=None)
    return p

def main() -> None:
    args = build_parser().parse_args()
    meta = make_sample(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        seed=args.seed,
        n_fraud=args.n_fraud,
        n_nonfraud=args.n_nonfraud,
        meta_json=Path(args.meta) if args.meta else None,
        step_min=args.step_min,
        step_max=args.step_max,
    )

    print("Sample created")
    print(f"- CSV  : {meta.output_path}")
    print(f"- Rows : {meta.total_rows} (fraud_rate={meta.fraud_rate:.4f})")
    print(f"- Step : [{meta.step_min}, {meta.step_max}]")
    if args.meta:
        print(f"- Meta : {args.meta}")

if __name__ == "__main__":
    main()