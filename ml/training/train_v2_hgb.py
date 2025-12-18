from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class TrainConfig:
    features_path: str
    step_col: str
    target_col: str
    split_method: str
    train_max_step: int
    test_min_step: int
    seed: int
    train_cap_rows: int
    drop_cols: List[str]
    model_name: str
    model_type: str
    model_params: Dict
    thresholds: List[float]
    model_dir: str
    report_md: str


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    features_path = cfg["data"]["features_path"]
    step_col = cfg["schema"]["step_col"]
    target_col = cfg["schema"]["target_col"]

    split = cfg["split"]
    split_method = str(split["method"])
    train_max_step = int(split["train_max_step"])
    test_min_step = int(split["test_min_step"])

    seed = int(cfg.get("repro", {}).get("seed", 42))

    training = cfg.get("training", {})
    train_cap_rows = int(training.get("train_cap_rows", 500_000))
    drop_cols = list(training.get("drop_cols", ["step"]))

    model = cfg["model"]
    model_name = str(model.get("name", "logreg_v1"))
    model_type = str(model.get("type", "logreg"))
    model_params = dict(model.get("params", {}))

    thresholds = list(cfg.get("thresholds", {}).get("report", [0.1, 0.2, 0.3, 0.5, 0.7]))

    artifacts = cfg.get("artifacts", {})
    model_dir = str(artifacts.get("model_dir", "ml/models/v1_baseline"))
    report_md = str(artifacts.get("report_md", "ml/reports/train_summary.md"))

    return TrainConfig(
        features_path=features_path,
        step_col=step_col,
        target_col=target_col,
        split_method=split_method,
        train_max_step=train_max_step,
        test_min_step=test_min_step,
        seed=seed,
        train_cap_rows=train_cap_rows,
        drop_cols=drop_cols,
        model_name=model_name,
        model_type=model_type,
        model_params=model_params,
        thresholds=thresholds,
        model_dir=model_dir,
        report_md=report_md,
    )


def temporal_split(
    df: pd.DataFrame,
    step_col: str,
    train_max_step: int,
    test_min_step: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if test_min_step != train_max_step + 1:
        raise ValueError(
            f"Temporal contract violated: test_min_step={test_min_step} must equal train_max_step+1={train_max_step + 1}"
        )
    train_df = df[df[step_col] <= train_max_step].copy()
    test_df = df[df[step_col] >= test_min_step].copy()
    if not train_df.empty and not test_df.empty:
        if train_df[step_col].max() >= test_df[step_col].min():
            raise AssertionError("Leakage detected: train overlaps test in step.")
    return train_df, test_df


def make_model(model_type: str, params: Dict):
    mt = str(model_type).lower()

    if mt == "logreg":
        if "random_state" not in params:
            params["random_state"] = 42
        return LogisticRegression(**params)

    if mt in {"hgb", "histgradientboosting", "histgradientboostingclassifier"}:
        allowed = {
            "loss", "learning_rate", "max_iter", "max_depth", "max_leaf_nodes",
            "min_samples_leaf", "l2_regularization", "max_bins", "categorical_features",
            "monotonic_cst", "interaction_cst", "warm_start", "early_stopping",
            "scoring", "validation_fraction", "n_iter_no_change", "tol",
            "verbose", "random_state"
        }
        clean_params = {k: v for k, v in params.items() if k in allowed}
        if "random_state" not in clean_params:
            clean_params["random_state"] = 42
        return HistGradientBoostingClassifier(**clean_params)

    raise ValueError(f"Unsupported model_type: {model_type}")


def file_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def write_report_md(
    out_path: str,
    model_flag: str,
    cfg: TrainConfig,
    n_all: int,
    n_train_full: int,
    n_train_used: int,
    n_test: int,
    step_train_max: int,
    step_test_min: int,
    pr_auc: float,
    th_table: List[Dict],
    cm_at_threshold: Dict,
    runtime_sec: float,
    model_size_bytes: int,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    def fmt_int(x: int) -> str:
        return f"{x:,}"

    def fmt_bytes(n: int) -> str:
        if n < 1024:
            return f"{n} B"
        if n < 1024**2:
            return f"{n/1024:.2f} KB"
        if n < 1024**3:
            return f"{n/1024**2:.2f} MB"
        return f"{n/1024**3:.2f} GB"

    title = "Baseline V1" if model_flag == "v1" else "Improved V2"

    md: List[str] = []
    md.append(f"# Train Summary — {title} ({cfg.model_type})\n\n")

    md.append("## Data Contract\n")
    md.append(f"- Split: temporal (anti-leakage)\n")
    md.append(f"- Train steps: ≤ {step_train_max}\n")
    md.append(f"- Test steps: ≥ {step_test_min}\n")
    md.append(f"- Dataset: `{cfg.features_path}`\n\n")

    md.append("## Data Size\n")
    md.append(f"- Total rows: {fmt_int(n_all)}\n")
    md.append(f"- Train rows (full split): {fmt_int(n_train_full)}\n")
    md.append(f"- Train rows used (cap): **{fmt_int(n_train_used)}**\n")
    md.append(f"- Test rows: {fmt_int(n_test)}\n\n")

    md.append("## Model\n")
    md.append(f"- Model type: {cfg.model_type}\n")
    md.append(f"- Model name: {cfg.model_name}\n")
    md.append(f"- Params: `{json.dumps(cfg.model_params, ensure_ascii=False)}`\n\n")

    md.append("- **Hyperparameters**:\n")
    for param, value in cfg.model_params.items():
        md.append(f"  - `{param}`: {value}\n")
    md.append("\n")

    md.append("## Metrics\n")
    md.append(f"- **PR-AUC (Average Precision)**: **{pr_auc:.6f}**\n\n")

    md.append("### Precision/Recall @ Thresholds\n")
    md.append("| Threshold | Precision | Recall |\n")
    md.append("|---:|---:|---:|\n")
    for row in th_table:
        md.append(f"| {row['threshold']:.2f} | {row['precision']:.6f} | {row['recall']:.6f} |\n")
    md.append("\n")

    md.append("### Confusion Matrix (selected threshold)\n")
    md.append(f"- Selected threshold: {cm_at_threshold['threshold']:.2f}\n\n")
    md.append("| | Pred 0 | Pred 1 |\n")
    md.append("|---|---:|---:|\n")
    md.append(f"| True 0 | {cm_at_threshold['tn']:,} | {cm_at_threshold['fp']:,} |\n")
    md.append(f"| True 1 | {cm_at_threshold['fn']:,} | {cm_at_threshold['tp']:,} |\n\n")

    md.append("## Engineering\n")
    md.append(f"- Training runtime: {runtime_sec:.2f} sec\n")
    md.append(f"- Model size: {fmt_bytes(model_size_bytes)}\n")

    Path(out_path).write_text("".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3.2/3.3 — Train V1/V2 (temporal split)")
    parser.add_argument("--config", type=str, default="ml/training/config.yaml")
    parser.add_argument("--model", type=str, default="v1", help="Model flag (use: v1 or v2)")
    args = parser.parse_args()

    model_flag = args.model.lower()
    if model_flag not in {"v1", "v2"}:
        raise ValueError("Supported --model: v1 (baseline), v2 (improved).")

    cfg = load_config(args.config)

    features_path = Path(cfg.features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_parquet(features_path)

    # Column checks
    for col in [cfg.step_col, cfg.target_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'. Available: {list(df.columns)[:30]}...")

    # Split
    if cfg.split_method.lower() != "temporal":
        raise ValueError(f"Only temporal split is supported. Got: {cfg.split_method}")

    train_df, test_df = temporal_split(df, cfg.step_col, cfg.train_max_step, cfg.test_min_step)

    # Cap training rows (after split)
    n_train_full = len(train_df)
    if cfg.train_cap_rows > 0 and len(train_df) > cfg.train_cap_rows:
        train_df = train_df.sort_values(cfg.step_col).tail(cfg.train_cap_rows)
    n_train_used = len(train_df)

    # Prepare X/y (drop non-feature cols)
    drop_cols = set(cfg.drop_cols + [cfg.target_col])
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df[cfg.target_col].astype(int)

    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    y_test = test_df[cfg.target_col].astype(int)

    # Fit model
    model = make_model(cfg.model_type, cfg.model_params)
    weights = np.where(y_train == 1, 20, 1)

    t0 = time.perf_counter()
    model.fit(X_train, y_train, sample_weight=weights)
    runtime_sec = time.perf_counter() - t0

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.predict(X_test)

    # Primary metric: PR-AUC
    pr_auc = float(average_precision_score(y_test, y_score))

    # Precision/Recall @ thresholds
    th_table: List[Dict] = []
    for th in cfg.thresholds:
        y_pred = (y_score >= float(th)).astype(int)
        th_table.append(
            {
                "threshold": float(th),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            }
        )

    # Confusion matrix at selected threshold
    selected_th = 0.5 if 0.5 in [float(t) for t in cfg.thresholds] else float(cfg.thresholds[0])
    y_pred_sel = (y_score >= selected_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_sel).ravel()
    cm_payload = {
        "threshold": float(selected_th),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # Save artifacts
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)

    metadata = {
        "model_flag": model_flag,
        "model_type": cfg.model_type,
        "model_name": cfg.model_name,
        "features_path": cfg.features_path,
        "target_col": cfg.target_col,
        "step_col": cfg.step_col,
        "split": {
            "method": cfg.split_method,
            "train_max_step": cfg.train_max_step,
            "test_min_step": cfg.test_min_step,
        },
        "train_cap_rows": cfg.train_cap_rows,
        "n_rows": {
            "all": int(len(df)),
            "train_full": int(n_train_full),
            "train_used": int(n_train_used),
            "test": int(len(test_df)),
        },
        "metrics": {
            "pr_auc": pr_auc,
            "threshold_table": th_table,
            "confusion_matrix": cm_payload,
        },
        "engineering": {
            "runtime_sec": float(runtime_sec),
            "model_size_bytes": int(file_size_bytes(model_path)),
        },
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_report_md(
        out_path=cfg.report_md,
        model_flag=model_flag,
        cfg=cfg,
        n_all=int(len(df)),
        n_train_full=int(n_train_full),
        n_train_used=int(n_train_used),
        n_test=int(len(test_df)),
        step_train_max=cfg.train_max_step,
        step_test_min=cfg.test_min_step,
        pr_auc=pr_auc,
        th_table=th_table,
        cm_at_threshold=cm_payload,
        runtime_sec=runtime_sec,
        model_size_bytes=int(file_size_bytes(model_path)),
    )

    print(f"[OK] {model_flag.upper()} trained. PR-AUC={pr_auc:.6f}")
    print(f"[OK] Saved model: {model_path}")
    print(f"[OK] Saved metadata: {model_dir / 'metadata.json'}")
    print(f"[OK] Wrote report: {cfg.report_md}")
    print(f"Train used: {n_train_used:,} (cap={cfg.train_cap_rows:,}) | Test: {len(test_df):,}")


if __name__ == "__main__":
    main()
