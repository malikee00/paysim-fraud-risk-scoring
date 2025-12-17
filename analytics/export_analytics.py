from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_kpis(input_parquet: Path, out_dir: Path) -> None:
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_parquet}")

    df = pd.read_parquet(input_parquet)

    # Expected canonical columns from transform.py
    required = {"step", "type", "amount", "isfraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for analytics export: {sorted(missing)}")

    # ---------- KPI 1: Daily (per step) ----------
    kpi_daily = (
        df.groupby("step", as_index=False)
        .agg(
            transactions=("step", "size"),
            fraud_transactions=("isfraud", "sum"),
            fraud_rate=("isfraud", "mean"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
        )
        .sort_values("step")
    )

    # ---------- KPI 2: By type ----------
    kpi_by_type = (
        df.groupby("type", as_index=False)
        .agg(
            transactions=("type", "size"),
            fraud_transactions=("isfraud", "sum"),
            fraud_rate=("isfraud", "mean"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
        )
        .sort_values("transactions", ascending=False)
    )

    ensure_parent(out_dir / "kpi_daily.csv")
    kpi_daily.to_csv(out_dir / "kpi_daily.csv", index=False)

    ensure_parent(out_dir / "kpi_by_type.csv")
    kpi_by_type.to_csv(out_dir / "kpi_by_type.csv", index=False)

    print(" Analytics exports created")
    print(f"- Input      : {input_parquet}")
    print(f"- Export dir : {out_dir}")
    print(f"- Files      : kpi_daily.csv, kpi_by_type.csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Export PowerBI-ready KPI CSVs from canonical dataset.")
    p.add_argument("--input", required=True, help="Input canonical parquet. e.g. data/processed/transactions_clean.parquet")
    p.add_argument("--out_dir", default="analytics/exports", help="Output directory for CSV exports.")
    args = p.parse_args()

    export_kpis(Path(args.input), Path(args.out_dir))


if __name__ == "__main__":
    main()
