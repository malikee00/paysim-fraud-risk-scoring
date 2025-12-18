from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


@dataclass(frozen=True)
class ThresholdPick:
    t1: float
    t2: float
    table: List[Dict]  
    pr_auc: float


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def threshold_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Iterable[float],
) -> List[Dict]:
    y_true = y_true.astype(int)
    rows: List[Dict] = []
    for th in thresholds:
        th = float(th)
        y_pred = (y_score >= th).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        pred_pos_rate = float(y_pred.mean())

        rows.append(
            {
                "threshold": th,
                "precision": float(precision),
                "recall": float(recall),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "pred_pos_rate": float(pred_pos_rate),
            }
        )
    return rows


def pick_t1_t2(
    y_true: np.ndarray,
    y_score: np.ndarray,
    candidate_thresholds: Optional[List[float]] = None,
    target_precision_T2: float = 0.995,
    min_recall_T2: float = 0.70,
    min_review_rate: float = 0.001,
    max_review_rate: float = 0.005,
    max_review_band_width: float = 0.25,   # NEW: force narrow band near T2
) -> ThresholdPick:
    """
    Pick T2 first (BLOCK) with high precision & enough recall.
    Then pick T1 close to T2 to keep REVIEW as a narrow band with controlled volume.
    """

    y_true = y_true.astype(int)
    pr_auc = float(average_precision_score(y_true, y_score))

    if not candidate_thresholds:
        candidate_thresholds = list(np.linspace(0.05, 0.95, 19))
    candidate_thresholds = sorted({float(t) for t in candidate_thresholds})

    table = threshold_table(y_true, y_score, candidate_thresholds)

    # --------------------------
    # 1) Choose T2 (BLOCK)
    # --------------------------
    feasible_T2 = [
        r for r in table
        if r["precision"] >= float(target_precision_T2) and r["recall"] >= float(min_recall_T2)
    ]
    if feasible_T2:
        t2 = float(sorted(feasible_T2, key=lambda r: r["threshold"])[0]["threshold"])
    else:
        best = sorted(table, key=lambda r: (r["precision"], r["recall"]), reverse=True)[0]
        t2 = float(best["threshold"])

    scores = y_score

    def review_rate(t1: float) -> float:
        return float(((scores >= t1) & (scores < t2)).mean())

    lower_bound = max(0.0, t2 - float(max_review_band_width))
    t1_candidates = [t for t in candidate_thresholds if lower_bound <= t < t2]

    if len(t1_candidates) < 3:
        extra = list(np.linspace(lower_bound, max(lower_bound, t2 - 1e-6), 30))
        t1_candidates = sorted({*(t1_candidates), *[float(x) for x in extra]})

    # --------------------------
    # 2) Choose T1 (REVIEW boundary) to meet review volume
    # Prefer the LARGEST T1 (closest to T2) that still gives enough review volume
    # --------------------------
    feasible = []
    for t1 in t1_candidates:
        rr = review_rate(t1)
        if rr >= float(min_review_rate) and rr <= float(max_review_rate):
            feasible.append((float(t1), float(rr)))

    if feasible:
        t1 = float(sorted(feasible, key=lambda x: x[0], reverse=True)[0][0])
    else:
        rr_at_lb = review_rate(lower_bound)
        if rr_at_lb < float(min_review_rate):
            below_t2 = scores[scores < t2]
            if below_t2.size == 0:
                t1 = float(max(0.0, t2 - 0.05))
            else:
                target_mass = float(min_review_rate)
                q = 1.0 - target_mass
                q = min(max(q, 0.0), 1.0)
                t1 = float(np.quantile(below_t2, q))
                t1 = float(min(t1, t2 - 1e-6))
                t1 = float(max(t1, lower_bound))
        else:
            candidates = [(float(t1), review_rate(t1)) for t1 in t1_candidates]
            under = [c for c in candidates if c[1] <= float(max_review_rate)]
            t1 = float(sorted(under, key=lambda x: x[0], reverse=True)[0][0]) if under else float(t2 - 1e-6)

    return ThresholdPick(t1=float(t1), t2=float(t2), table=table, pr_auc=pr_auc)


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), y_score)
    return precision, recall, thresholds
