from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ErrorCaseConfig:
    top_k: int = 5
    amount_col: Optional[str] = None
    id_cols: Optional[List[str]] = None  
    score_col_name: str = "_score"
    pred_bucket_col_name: str = "_bucket"


def _ensure_parent(p: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def _pick_cols_for_display(df: pd.DataFrame, cfg: ErrorCaseConfig) -> List[str]:
    cols = []
    if cfg.id_cols:
        for c in cfg.id_cols:
            if c in df.columns:
                cols.append(c)

    for c in ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "nameOrig", "nameDest"]:
        if c in df.columns and c not in cols:
            cols.append(c)

    if cfg.amount_col and cfg.amount_col in df.columns and cfg.amount_col not in cols:
        cols.append(cfg.amount_col)

    if cfg.score_col_name in df.columns:
        cols.append(cfg.score_col_name)
    if cfg.pred_bucket_col_name in df.columns:
        cols.append(cfg.pred_bucket_col_name)

    return cols


def assign_bucket_from_thresholds(score: np.ndarray, t1: float, t2: float) -> np.ndarray:
    b = np.zeros_like(score, dtype=int)
    b[(score >= t1) & (score < t2)] = 1
    b[score >= t2] = 2
    return b


def extract_error_cases(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    t1: float,
    t2: float,
    cfg: Optional[ErrorCaseConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict with:
      - fn_approve: fraud (1) but APPROVED (score < t1) => worst FN
      - fp_block: legit (0) but BLOCKED (score >= t2) => worst FP
    """
    cfg = cfg or ErrorCaseConfig()

    out = df_test.copy()
    out[cfg.score_col_name] = y_score
    out["_y_true"] = y_true.astype(int)
    out[cfg.pred_bucket_col_name] = assign_bucket_from_thresholds(y_score, t1, t2)

    # FN approve
    fn_approve = out[(out["_y_true"] == 1) & (out[cfg.pred_bucket_col_name] == 0)].copy()
    sort_cols = [cfg.score_col_name]
    ascending = [False]
    if cfg.amount_col and cfg.amount_col in fn_approve.columns:
        sort_cols = [cfg.amount_col, cfg.score_col_name]
        ascending = [False, False]
    fn_approve = fn_approve.sort_values(sort_cols, ascending=ascending).head(cfg.top_k)

    # FP block
    fp_block = out[(out["_y_true"] == 0) & (out[cfg.pred_bucket_col_name] == 2)].copy()
    sort_cols = [cfg.score_col_name]
    ascending = [False]
    if cfg.amount_col and cfg.amount_col in fp_block.columns:
        sort_cols = [cfg.amount_col, cfg.score_col_name]
        ascending = [False, False]
    fp_block = fp_block.sort_values(sort_cols, ascending=ascending).head(cfg.top_k)

    display_cols = _pick_cols_for_display(out, cfg)

    return {
        "fn_approve": fn_approve[display_cols] if display_cols else fn_approve,
        "fp_block": fp_block[display_cols] if display_cols else fp_block,
    }


def write_error_cases_csv(cases: Dict[str, pd.DataFrame], out_csv: str) -> None:
    _ensure_parent(out_csv)
    frames = []
    for name, df in cases.items():
        tmp = df.copy()
        tmp["_case_type"] = name
        frames.append(tmp)
    if frames:
        pd.concat(frames, axis=0).to_csv(out_csv, index=False)
    else:
        pd.DataFrame().to_csv(out_csv, index=False)


def _hypothesis_and_actions_template() -> List[Dict[str, str]]:
    return [
        {
            "hypothesis": "FN approve terjadi karena fitur tidak menangkap pola 'burst/velocity' (fraud cepat beruntun) atau pola hubungan origin-destination.",
            "action": "Tambah fitur velocity (tx_count_recent, amount_sum_recent), unique_counterparty_recent, serta delta balance ratio."
        },
        {
            "hypothesis": "FP block terjadi pada transaksi legitimate dengan amount tinggi yang mirip pola fraud (mis. payroll/transfer rutin).",
            "action": "Tambah fitur 'behavioral baseline' per user: typical_amount, zscore_amount_user, schedule periodicity; atau whitelist segment tertentu."
        },
        {
            "hypothesis": "Score calibration tidak stabil: model sangat confident di area high-score sehingga beberapa legit ikut keblok.",
            "action": "Coba calibration (isotonic/Platt) untuk score, atau tuning T2 berbasis 'max acceptable FP per day'."
        },
        {
            "hypothesis": "Missing segment columns (type/amount bucket/activity) bikin blind spot untuk error di segmen spesifik.",
            "action": "Pastikan kolom segment tersedia di features parquet (type, amount) dan generate activity proxy (tx_count_recent) untuk evaluasi segment-level."
        },
    ]
