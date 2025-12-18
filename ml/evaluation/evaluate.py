from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from ml.evaluation.thresholding import pick_t1_t2, pr_curve_points, threshold_table
from ml.evaluation.error_analysis import ErrorCaseConfig, extract_error_cases, write_error_cases_csv


@dataclass(frozen=True)
class EvalConfig:
    features_path: str
    step_col: str
    target_col: str
    train_max_step: int
    test_min_step: int
    drop_cols: List[str]

    model_dir: str

    pr_curve_png: str
    thresholds_yaml: str
    eval_report_md: str
    candidate_thresholds: List[float]

    target_precision_T2: float
    min_recall_T2: float
    min_review_rate: float
    max_review_rate: float
    max_review_band_width: float

    seg_by_type_col: Optional[str]
    seg_amount_col: Optional[str]
    seg_amount_bins: Optional[List[float]]
    seg_user_activity_col: Optional[str]
    seg_top_k: int

    error_cases_csv: str
    error_top_k: int
    error_amount_col: Optional[str]
    error_id_cols: Optional[List[str]]



def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_eval_config(path: str) -> EvalConfig:
    cfg = load_yaml(path)

    features_path = cfg["data"]["features_path"]
    step_col = cfg["schema"]["step_col"]
    target_col = cfg["schema"]["target_col"]
    train_max_step = int(cfg["split"]["train_max_step"])
    test_min_step = int(cfg["split"]["test_min_step"])
    drop_cols = list(cfg.get("training", {}).get("drop_cols", ["step"]))

    model_dir = str(cfg.get("artifacts", {}).get("model_dir", "ml/models/model"))

    ev = cfg.get("evaluation", {}) or {}
    pr_curve_png = str(ev.get("pr_curve_png", "ml/reports/pr_curve.png"))
    thresholds_yaml = str(ev.get("thresholds_yaml", "ml/reports/thresholds.yaml"))
    eval_report_md = str(ev.get("eval_report_md", "ml/reports/eval_report.md"))
    error_cases_csv = str(ev.get("error_cases_csv", "ml/reports/error_cases.csv"))
    error_top_k = int(ev.get("error_top_k", 5))
    error_amount_col = ev.get("error_amount_col", None)
    error_id_cols = ev.get("error_id_cols", None)

    candidate_thresholds = list(ev.get("candidate_thresholds", []))

    pol = ev.get("policy", {}) or {}
    max_review_band_width = float(pol.get("max_review_band_width", 0.25))

    target_precision_T2 = float(pol.get("target_precision_T2", 0.995))
    min_recall_T2 = float(pol.get("min_recall_T2", 0.70))
    min_review_rate = float(pol.get("min_review_rate", 0.001))
    max_review_rate = float(pol.get("max_review_rate", 0.05))

    seg = ev.get("segments", {}) or {}
    by_type_col = seg.get("by_type_col", None)
    amount_col = seg.get("amount_col", None)
    amount_bins = seg.get("amount_bins", None)
    user_activity_col = seg.get("user_activity_col", None)
    top_k = int(seg.get("top_k", 5))

    return EvalConfig(
        features_path=features_path,
        step_col=step_col,
        target_col=target_col,
        train_max_step=train_max_step,
        test_min_step=test_min_step,
        drop_cols=drop_cols,
        model_dir=model_dir,
        pr_curve_png=pr_curve_png,
        thresholds_yaml=thresholds_yaml,
        eval_report_md=eval_report_md,
        candidate_thresholds=candidate_thresholds,
        target_precision_T2=target_precision_T2,
        min_recall_T2=min_recall_T2,
        min_review_rate=min_review_rate,
        max_review_rate=max_review_rate,
        max_review_band_width=max_review_band_width,
        seg_by_type_col=by_type_col,
        seg_amount_col=amount_col,
        seg_amount_bins=amount_bins,
        seg_user_activity_col=user_activity_col,
        seg_top_k=top_k,
        error_cases_csv=error_cases_csv,
        error_top_k=error_top_k,
        error_amount_col=error_amount_col,
        error_id_cols=error_id_cols,

    )


def temporal_test_split(df: pd.DataFrame, step_col: str, test_min_step: int) -> pd.DataFrame:
    return df[df[step_col] >= test_min_step].copy()


def score_model(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_pr_curve_png(y_true: np.ndarray, y_score: np.ndarray, out_png: str) -> None:
    precision, recall, _ = pr_curve_points(y_true, y_score)
    ensure_parent(out_png)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def assign_bucket(score: np.ndarray, t1: float, t2: float) -> np.ndarray:

    b = np.zeros_like(score, dtype=int)
    b[(score >= t1) & (score < t2)] = 1
    b[score >= t2] = 2
    return b


def bucket_metrics(y_true: np.ndarray, y_score: np.ndarray, t1: float, t2: float) -> Dict:
    """
    Buckets:
      - Low  (approve): score < t1
      - Med  (review):  t1 <= score < t2
      - High (block):   score >= t2

    Metrics (as per design):
      - Low: FP rate (legit flagged to review/block) = FPR at t1
      - Med: precision & volume (fraud rate within review bucket)
      - High: recall fraud (share of all fraud caught in high bucket)
    """
    y_true = y_true.astype(int)
    total = len(y_true)

    low_mask = y_score < t1
    med_mask = (y_score >= t1) & (y_score < t2)
    high_mask = y_score >= t2

    # volumes
    low_vol = int(low_mask.sum())
    med_vol = int(med_mask.sum())
    high_vol = int(high_mask.sum())

    low_rate = float(low_vol / total) if total else 0.0
    med_rate = float(med_vol / total) if total else 0.0
    high_rate = float(high_vol / total) if total else 0.0

    # --- Low bucket FP rate (FPR @ T1): among legit (y=0), how many are flagged (score>=t1)
    legit_mask = y_true == 0
    legit_total = int(legit_mask.sum())
    legit_flagged = int(((y_score >= t1) & legit_mask).sum())  
    fp_rate_at_t1 = float(legit_flagged / legit_total) if legit_total else 0.0

    # --- Medium precision: P(fraud | review bucket)
    med_total = med_vol
    med_fraud = int(((y_true == 1) & med_mask).sum())
    precision_review = float(med_fraud / med_total) if med_total else 0.0

    # --- High fraud recall: P(score>=T2 | fraud)
    fraud_total = int((y_true == 1).sum())
    fraud_in_high = int(((y_true == 1) & high_mask).sum())
    fraud_recall_in_high = float(fraud_in_high / fraud_total) if fraud_total else 0.0

    return {
        "low": {
            "volume": low_vol,
            "volume_rate": low_rate,
            "fp_rate_flagged_at_T1": fp_rate_at_t1,
        },
        "medium": {
            "volume": med_vol,
            "volume_rate": med_rate,
            "precision_in_review": precision_review,
        },
        "high": {
            "volume": high_vol,
            "volume_rate": high_rate,
            "fraud_recall_in_high": fraud_recall_in_high,
        },
        "rates": {
            "approve_rate": low_rate,
            "review_rate": med_rate,
            "block_rate": high_rate,
        },
    }


def worst_segments_table(
    df_test: pd.DataFrame,
    y_score: np.ndarray,
    target_col: str,
    segment_col: str,
    metric: str,
    top_k: int,
) -> List[Dict]:
    # metric choices: "precision", "recall", "fraud_rate", "avg_score"
    if segment_col not in df_test.columns:
        return []

    tmp = df_test[[segment_col, target_col]].copy()
    tmp["_score"] = y_score

    rows: List[Dict] = []
    for seg_val, g in tmp.groupby(segment_col, dropna=False):
        yt = g[target_col].astype(int).to_numpy()
        sc = g["_score"].to_numpy()
        n = len(g)
        fraud_rate = float(yt.mean()) if n else 0.0
        avg_score = float(sc.mean()) if n else 0.0

        th = 0.5
        yp = (sc >= th).astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0

        rows.append(
            {
                "segment": str(seg_val),
                "n": int(n),
                "fraud_rate": fraud_rate,
                "avg_score": avg_score,
                "precision@0.50": precision,
                "recall@0.50": recall,
            }
        )

    # rank "worst" depending on metric
    if metric == "precision":
        rows = sorted(rows, key=lambda r: (r["precision@0.50"], -r["n"]))
    elif metric == "recall":
        rows = sorted(rows, key=lambda r: (r["recall@0.50"], -r["n"]))
    elif metric == "fraud_rate":
        rows = sorted(rows, key=lambda r: (-r["fraud_rate"], -r["n"]))
    else:  # avg_score
        rows = sorted(rows, key=lambda r: (-r["avg_score"], -r["n"]))

    return rows[:top_k]


def amount_bucket_col(amount: pd.Series, bins: List[float]) -> pd.Series:
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"{bins[i]:g}-{bins[i+1]:g}")
    return pd.cut(amount, bins=bins, labels=labels, include_lowest=True)


def write_eval_report_md(
    out_path: str,
    pr_auc: float,
    t1: float,
    t2: float,
    th_table: List[Dict],
    bucket_perf: Dict,
    segments: Dict[str, List[Dict]],
    error_cases: Dict[str, pd.DataFrame],
    error_cases_csv_path: Optional[str] = None,
) -> None:

    md: List[str] = []
    md.append("# Evaluation Report\n\n")

    md.append("## A) Global Metrics\n")
    md.append(f"- **PR-AUC (primary)**: **{pr_auc:.6f}**\n\n")

    md.append("### Precision/Recall @ Candidate Thresholds\n")
    md.append("| Threshold | Precision | Recall | Pred_Pos_Rate |\n")
    md.append("|---:|---:|---:|---:|\n")
    for r in th_table:
        md.append(f"| {r['threshold']:.2f} | {r['precision']:.6f} | {r['recall']:.6f} | {r['pred_pos_rate']:.6f} |\n")
    md.append("\n")

    md.append("## B) Threshold Selection (T1/T2)\n")
    md.append(f"- **T1 (approve→review)**: **{t1:.2f}**\n")
    md.append(f"- **T2 (review→block)**: **{t2:.2f}**\n\n")
    md.append("Decision mapping:\n")
    md.append("- score < T1 → **APPROVE (Low risk)**\n")
    md.append("- T1 ≤ score < T2 → **REVIEW (Medium risk)**\n")
    md.append("- score ≥ T2 → **BLOCK (High risk)**\n\n")

    md.append("## C) Bucket Performance\n")
    md.append("| Bucket | Volume | Volume Rate | Key Metric |\n")
    md.append("|---|---:|---:|---|\n")
    low = bucket_perf.get("low", {})
    med = bucket_perf.get("medium", {})
    high = bucket_perf.get("high", {})
    rates = bucket_perf.get("rates", {})

    md.append(
        f"| Low | {low.get('volume',0)} | {low.get('volume_rate',0.0):.6f} | "
        f"fp_rate_flagged_at_T1={low.get('fp_rate_flagged_at_T1',0.0):.6f} |\n"
    )
    md.append(
        f"| Medium | {med.get('volume',0)} | {med.get('volume_rate',0.0):.6f} | "
        f"precision_in_review={med.get('precision_in_review',0.0):.6f} |\n"
    )
    md.append(
        f"| High | {high.get('volume',0)} | {high.get('volume_rate',0.0):.6f} | "
        f"fraud_recall_in_high={high.get('fraud_recall_in_high',0.0):.6f} |\n"
    )
    md.append("\n")

    md.append("### Bucket Rates\n")
    md.append(f"- approve_rate: {rates.get('approve_rate',0.0):.6f}\n")
    md.append(f"- review_rate: {rates.get('review_rate',0.0):.6f}\n")
    md.append(f"- block_rate: {rates.get('block_rate',0.0):.6f}\n\n")

    md.append("\n")

    md.append("## Error Analysis\n")
    md.append("Fokus: (1) **False Negative yang lolos (APPROVE)** dan (2) **False Positive yang keblok (BLOCK)**.\n\n")

    fn_df = error_cases.get("fn_approve")
    fp_df = error_cases.get("fp_block")

    md.append("### False Negatives — Fraud APPROVED (worst-case)\n")
    if fn_df is None or fn_df.empty:
        md.append("_No FN approve cases found (or columns not available)._ \n\n")
    else:
        md.append(f"- Showing top {len(fn_df)} cases (prioritize high amount / high score)\n\n")
        md.append(fn_df.to_markdown(index=False))
        md.append("\n\n")

    md.append("### False Positives — Legit BLOCKED (most painful)\n")
    if fp_df is None or fp_df.empty:
        md.append("_No FP block cases found (or columns not available)._ \n\n")
    else:
        md.append(f"- Showing top {len(fp_df)} cases (prioritize high amount / high score)\n\n")
        md.append(fp_df.to_markdown(index=False))
        md.append("\n\n")

    md.append("### Hypotheses & Next Actions (3–5 bullets)\n")
    bullets = [
        "FN approve cenderung terjadi pada pola velocity/sequence yang belum tertangkap (butuh fitur tx_count_recent / amount_sum_recent).",
        "FP block kemungkinan pada transaksi legitimate beramount tinggi yang mirip fraud; butuh fitur baseline perilaku per user (typical_amount / zscore).",
        "Pertimbangkan score calibration (Platt/Isotonic) agar confidence lebih stabil sebelum menetapkan T2 sangat tinggi.",
        "Tambahkan evaluasi per segmen (type, amount bucket, user activity) untuk menemukan blind spot spesifik segmen.",
    ]
    for b in bullets[:5]:
        md.append(f"- {b}\n")
    md.append("\n")

    if error_cases_csv_path:
        md.append(f"_(Optional) Full cases exported to `{Path(error_cases_csv_path).as_posix()}`_\n\n")



    md.append("## D) Segment-level Evaluation (Top worst)\n")
    for title, rows in segments.items():
        md.append(f"### {title}\n")
        if not rows:
            md.append("_Skipped (segment column not available)_\n\n")
            continue
        md.append("| Segment | n | fraud_rate | avg_score | precision@0.50 | recall@0.50 |\n")
        md.append("|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            md.append(
                f"| {r['segment']} | {r['n']} | {r['fraud_rate']:.6f} | {r['avg_score']:.6f} | {r['precision@0.50']:.6f} | {r['recall@0.50']:.6f} |\n"
            )
        md.append("\n")

    Path(out_path).write_text("".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3.5 — Evaluation strategy implementation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_flag", type=str, default="v2") 
    args = parser.parse_args()

    cfg = load_eval_config(args.config)

    df = pd.read_parquet(cfg.features_path)
    if cfg.step_col not in df.columns or cfg.target_col not in df.columns:
        raise KeyError(f"Missing required columns. Need: {cfg.step_col}, {cfg.target_col}")

    df_test = temporal_test_split(df, cfg.step_col, cfg.test_min_step)
    if df_test.empty:
        raise ValueError("Test split is empty. Check test_min_step / data.")

    drop_cols = set(cfg.drop_cols + [cfg.target_col])
    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])
    y_test = df_test[cfg.target_col].astype(int).to_numpy()

    model_path = Path(cfg.model_dir) / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)

    y_score = score_model(model, X_test)

    # A) Global
    pr_auc = float(average_precision_score(y_test, y_score))

    # B) Threshold selection
    pick = pick_t1_t2(
        y_true=y_test,
        y_score=y_score,
        candidate_thresholds=cfg.candidate_thresholds,
        target_precision_T2=cfg.target_precision_T2,
        min_recall_T2=cfg.min_recall_T2,
        min_review_rate=cfg.min_review_rate,
        max_review_rate=cfg.max_review_rate,
        max_review_band_width=cfg.max_review_band_width,
    )
    t1, t2 = pick.t1, pick.t2
    th_table = pick.table

    # Save thresholds.yaml
    ensure_parent(cfg.thresholds_yaml)
    payload = {
        "t1": float(t1),
        "t2": float(t2),
        "policy": {
            "target_precision_T2": cfg.target_precision_T2,
            "min_recall_T2": cfg.min_recall_T2,
            "min_review_rate": cfg.min_review_rate,
            "max_review_rate": cfg.max_review_rate,
            "max_review_band_width": cfg.max_review_band_width
        },
        "candidates": [float(r["threshold"]) for r in th_table],
    }
    with open(cfg.thresholds_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    # PR curve
    save_pr_curve_png(y_test, y_score, cfg.pr_curve_png)

    # C) Bucket eval
    bucket_perf = bucket_metrics(y_test, y_score, t1, t2)

    cases = extract_error_cases(
        df_test=df_test,
        y_true=y_test,
        y_score=y_score,
        t1=t1,
        t2=t2,
        cfg=ErrorCaseConfig(
            top_k=cfg.error_top_k,
            amount_col=cfg.error_amount_col,
            id_cols=cfg.error_id_cols,
        ),
    )
    write_error_cases_csv(cases, cfg.error_cases_csv)


    # D) Segment eval
    segments: Dict[str, List[Dict]] = {}

    # by type
    if cfg.seg_by_type_col and cfg.seg_by_type_col in df_test.columns:
        segments["By type"] = worst_segments_table(df_test, y_score, cfg.target_col, cfg.seg_by_type_col, "precision", cfg.seg_top_k)
    else:
        segments["By type"] = []

    # by amount bucket
    if cfg.seg_amount_col and cfg.seg_amount_bins and cfg.seg_amount_col in df_test.columns:
        df_amt = df_test.copy()
        df_amt["_amount_bucket"] = amount_bucket_col(df_amt[cfg.seg_amount_col], [float(x) for x in cfg.seg_amount_bins])
        segments["By amount bucket"] = worst_segments_table(df_amt, y_score, cfg.target_col, "_amount_bucket", "precision", cfg.seg_top_k)
    else:
        segments["By amount bucket"] = []

    # by user activity proxy
    if cfg.seg_user_activity_col and cfg.seg_user_activity_col in df_test.columns:
        # bucketize into quantiles for stability
        s = df_test[cfg.seg_user_activity_col]
        try:
            q = pd.qcut(s, q=3, labels=["low", "mid", "high"], duplicates="drop")
            df_act = df_test.copy()
            df_act["_activity_bucket"] = q.astype(str)
            segments["By user activity proxy"] = worst_segments_table(df_act, y_score, cfg.target_col, "_activity_bucket", "precision", cfg.seg_top_k)
        except Exception:
            segments["By user activity proxy"] = []
    else:
        segments["By user activity proxy"] = []

    # Report
    write_eval_report_md(
    cfg.eval_report_md,
    pr_auc,
    t1,
    t2,
    th_table,
    bucket_perf,
    segments,
    cases,
    error_cases_csv_path=cfg.error_cases_csv,
)

    print("[OK] Evaluation done")
    print(f"[OK] PR-AUC={pr_auc:.6f}")
    print(f"[OK] Saved thresholds: {cfg.thresholds_yaml}")
    print(f"[OK] Saved PR curve: {cfg.pr_curve_png}")
    print(f"[OK] Wrote report: {cfg.eval_report_md}")


if __name__ == "__main__":
    main()