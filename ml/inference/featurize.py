from __future__ import annotations

import math
from typing import Any, Dict, List


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        return str(x)
    except Exception:
        return default


def _log1p_nonneg(x: float) -> float:
    x = max(0.0, float(x))
    return float(math.log1p(x))


def raw_to_features(raw: Dict[str, Any], *, required_features: List[str]) -> Dict[str, float]:
    """
    Convert raw form payload into the 18 engineered features expected by the v2 model.

    raw expected keys (minimal):
      - step, type, amount
      - oldbalanceOrg, newbalanceOrig
      - oldbalanceDest, newbalanceDest
      - optional: nameOrig, nameDest

    Notes:
      - Some features require historical aggregates (window counts/means, uniq origins hist).
        For demo we set safe defaults so API works reliably.
      - This is production-friendly: later you can replace defaults by pulling from a realtime store.
    """

    step = _safe_int(raw.get("step"), 0)
    tx_type = _safe_str(raw.get("type"), "TRANSFER").upper()
    amount = _safe_float(raw.get("amount"), 0.0)

    old_org = _safe_float(raw.get("oldbalanceOrg"), 0.0)
    new_org = _safe_float(raw.get("newbalanceOrig"), 0.0)
    old_dst = _safe_float(raw.get("oldbalanceDest"), 0.0)
    new_dst = _safe_float(raw.get("newbalanceDest"), 0.0)

    name_dest = _safe_str(raw.get("nameDest"), "")
    # name_orig = _safe_str(raw.get("nameOrig"), "")

    log_amount = _log1p_nonneg(amount)

    org_actual_delta = new_org - old_org
    dst_actual_delta = new_dst - old_dst

    org_expected_delta = -amount
    dst_expected_delta = amount

    org_delta_mismatch = org_actual_delta - org_expected_delta
    dest_delta_mismatch = dst_actual_delta - dst_expected_delta

    error_orig = abs(org_delta_mismatch)
    error_dest = abs(dest_delta_mismatch)

    org_balance_decreased = 1.0 if new_org < old_org else 0.0
    dest_balance_increased = 1.0 if new_dst > old_dst else 0.0

    is_merchant_dest = 1.0 if name_dest.upper().startswith("M") else 0.0

    # ---------- type one-hot ----------
    type_cash_in = 1.0 if tx_type == "CASH_IN" else 0.0
    type_cash_out = 1.0 if tx_type == "CASH_OUT" else 0.0
    type_debit = 1.0 if tx_type == "DEBIT" else 0.0
    type_payment = 1.0 if tx_type == "PAYMENT" else 0.0
    type_transfer = 1.0 if tx_type == "TRANSFER" else 0.0

    # ---------- historical/rolling features (demo-safe defaults) ----------
    # In real production you compute these from a store keyed by nameDest:
    # - dest_txn_count_w10, dest_txn_count_w50
    # - ratio_amt_to_dest_mean_w10, ratio_amt_to_dest_mean_w50
    # - dest_uniq_origs_hist
    #
    # For demo: set counts = 0; ratios = 1.0 (neutral) to avoid divide-by-zero weirdness.
    dest_uniq_origs_hist = 0.0
    dest_txn_count_w10 = 0.0
    dest_txn_count_w50 = 0.0
    ratio_amt_to_dest_mean_w10 = 1.0
    ratio_amt_to_dest_mean_w50 = 1.0

    computed = {
        "log_amount": log_amount,
        "error_orig": error_orig,
        "error_dest": error_dest,
        "org_delta_mismatch": org_delta_mismatch,
        "dest_delta_mismatch": dest_delta_mismatch,
        "is_merchant_dest": is_merchant_dest,
        "org_balance_decreased": org_balance_decreased,
        "dest_balance_increased": dest_balance_increased,
        "dest_uniq_origs_hist": dest_uniq_origs_hist,
        "type_CASH_IN": type_cash_in,
        "type_CASH_OUT": type_cash_out,
        "type_DEBIT": type_debit,
        "type_PAYMENT": type_payment,
        "type_TRANSFER": type_transfer,
        "dest_txn_count_w10": dest_txn_count_w10,
        "ratio_amt_to_dest_mean_w10": ratio_amt_to_dest_mean_w10,
        "dest_txn_count_w50": dest_txn_count_w50,
        "ratio_amt_to_dest_mean_w50": ratio_amt_to_dest_mean_w50,
    }

    out: Dict[str, float] = {}
    for f in required_features:
        out[f] = float(computed.get(f, 0.0))
    return out
