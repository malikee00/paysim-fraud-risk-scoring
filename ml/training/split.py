from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import yaml


@dataclass(frozen=True)
class Config:
    features_path: str
    step_col: str
    target_col: str
    split_method: str
    train_max_step: int
    test_min_step: int
    seed: int
    report_path: str


def load_config(config_path: str) -> Config:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    features_path = cfg["data"]["features_path"]
    step_col = cfg["schema"]["step_col"]
    target_col = cfg["schema"]["target_col"]

    split = cfg["split"]
    split_method = str(split["method"])
    train_max_step = int(split["train_max_step"])
    test_min_step = int(split["test_min_step"])

    seed = int(cfg.get("repro", {}).get("seed", 42))
    report_path = str(cfg.get("reports", {}).get("data_split_md", "ml/reports/data_split.md"))

    return Config(
        features_path=features_path,
        step_col=step_col,
        target_col=target_col,
        split_method=split_method,
        train_max_step=train_max_step,
        test_min_step=test_min_step,
        seed=seed,
        report_path=report_path,
    )


def temporal_split(
    df: pd.DataFrame,
    step_col: str,
    train_max_step: int,
    test_min_step: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Contract check: anti-leakage
    if test_min_step != train_max_step + 1:
        raise ValueError(
            f"Temporal contract violated: test_min_step={test_min_step} "
            f"must equal train_max_step+1={train_max_step + 1}."
        )

    if step_col not in df.columns:
        raise KeyError(f"Missing step column '{step_col}' in dataset.")

    train_df = df[df[step_col] <= train_max_step].copy()
    test_df = df[df[step_col] >= test_min_step].copy()

    # Additional guard: ensure no overlap
    if not train_df.empty and not test_df.empty:
        max_train = train_df[step_col].max()
        min_test = test_df[step_col].min()
        if max_train >= min_test:
            raise AssertionError(
                f"Leakage detected: max(train.{step_col})={max_train} "
                f">= min(test.{step_col})={min_test}."
            )

    return train_df, test_df


def safe_mean(series: pd.Series) -> Optional[float]:
    try:
        return float(series.mean())
    except Exception:
        return None


def write_report(
    report_path: str,
    df_all: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    step_col: str,
    target_col: str,
    train_max_step: int,
    test_min_step: int,
) -> None:
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    all_min, all_max = int(df_all[step_col].min()), int(df_all[step_col].max())

    train_min = int(train_df[step_col].min()) if not train_df.empty else None
    train_max = int(train_df[step_col].max()) if not train_df.empty else None
    test_min = int(test_df[step_col].min()) if not test_df.empty else None
    test_max = int(test_df[step_col].max()) if not test_df.empty else None

    all_rate = safe_mean(df_all[target_col]) if target_col in df_all.columns else None
    train_rate = safe_mean(train_df[target_col]) if target_col in train_df.columns else None
    test_rate = safe_mean(test_df[target_col]) if target_col in test_df.columns else None

    def fmt_int(x: int) -> str:
        return f"{x:,}"

    def fmt_rate(x: Optional[float]) -> str:
        return "N/A" if x is None else f"{x:.6f}"

    md = []
    md.append("# Data Split Report (Temporal)\n\n")

    md.append("## Contract\n")
    md.append("- split.method: temporal\n")
    md.append(f"- train_max_step (T): {train_max_step}\n")
    md.append(f"- test_min_step (T+1): {test_min_step}\n\n")

    md.append("## Dataset Overview\n")
    md.append(f"- Total rows: {fmt_int(len(df_all))}\n")
    md.append(f"- Step range: {all_min} → {all_max}\n")
    if all_rate is not None:
        md.append(f"- Overall fraud rate (mean {target_col}): {fmt_rate(all_rate)}\n")
    md.append("\n")

    md.append("## Split Summary\n")
    md.append("| Split | Rows | Step Range | Fraud Rate |\n")
    md.append("|---|---:|---|---:|\n")
    md.append(
        f"| Train | {fmt_int(len(train_df))} | {train_min} → {train_max} | {fmt_rate(train_rate)} |\n"
    )
    md.append(
        f"| Test | {fmt_int(len(test_df))} | {test_min} → {test_max} | {fmt_rate(test_rate)} |\n"
    )
    md.append("\n")

    md.append("## Sanity Checks\n")
    md.append(f" No overlap: max(train.{step_col})={train_max} < min(test.{step_col})={test_min}\n")
    md.append(" Temporal split (anti-leakage): train uses past steps only\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(md))


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3.1 Temporal split (anti-leakage)")
    parser.add_argument("--config", type=str, default="ml/training/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if cfg.split_method.lower() != "temporal":
        raise ValueError(f"Only temporal split is supported here. Got: {cfg.split_method}")

    features_path = Path(cfg.features_path)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}\n"
            "Expected: data/processed_local/features_v1_full.parquet"
        )

    df = pd.read_parquet(features_path)

    # minimal schema checks
    for col in [cfg.step_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'. Columns: {list(df.columns)[:30]}...")

    train_df, test_df = temporal_split(
        df=df,
        step_col=cfg.step_col,
        train_max_step=cfg.train_max_step,
        test_min_step=cfg.test_min_step,
    )

    write_report(
        report_path=cfg.report_path,
        df_all=df,
        train_df=train_df,
        test_df=test_df,
        step_col=cfg.step_col,
        target_col=cfg.target_col,
        train_max_step=cfg.train_max_step,
        test_min_step=cfg.test_min_step,
    )

    print(f"[OK] Wrote report: {cfg.report_path}")
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    if not train_df.empty and not test_df.empty:
        print(f"Train step <= {cfg.train_max_step} | Test step >= {cfg.test_min_step}")


if __name__ == "__main__":
    main()