# Model Registry

## Current
- **current_version**: **v2**
- **current_model_dir**: `ml/models/v2_improved`
- **thresholds_config**: `ml/reports/thresholds.yaml`
- **eval_report**: `ml/reports/eval_report.md`

---

## v1 — Baseline (logreg)
- **model_dir**: `ml/models/v1_baseline`
- **train_summary**: `ml/reports/train_summary.md`
- **key metrics**:
  - PR-AUC: 0.733998
  - Threshold used in train summary: 0.50
  - Notes: baseline for comparison

---

## v2 — Improved (hgb)
- **model_dir**: `ml/models/v2_improved`
- **train_summary**: `ml/reports/train_summary_v2.md`
- **eval report**: `ml/reports/eval_report.md`
- **key metrics**:
  - PR-AUC: 0.899141
  - Selected thresholds (policy-driven):
    - T1 (approve→review): 0.50
    - T2 (review→block): 0.85
  - Bucket performance (test):
    - approve_rate: 0.994573
    - review_rate: 0.000584
    - block_rate: 0.004844
    - precision_in_review: 0.541667
    - fraud_recall_in_high: 0.742773

---

## Change Log
- 2025-12-xx: Set `current = v2` based on PR-AUC + bucket metrics + error analysis.
