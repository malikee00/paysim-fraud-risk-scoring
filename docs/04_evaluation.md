# Evaluation & Decision Logic

## Context
Model evaluation is designed to reflect **operational decision-making**, not just offline predictive performance.

Evaluation is performed at both the **probability level** and the **decision level**.

---

## Key Decisions
- Primary metric:
  - PR-AUC
- Secondary metrics:
  - Precision / Recall at decision thresholds
- Decision outputs mapped to:
  - approve / review / block
- Segment-level evaluation by:
  - transaction type
  - amount bucket

---

## Evidence / Artefacts
- Evaluation logic:  
  `ml/evaluation/eval.py`
- Threshold configuration:  
  `ml/reports/thresholds.yaml`
- Evaluation report:  
  `ml/reports/eval_report.md`

---

## Trade-offs / Assumptions
- Thresholds are proxy-based, not cost-optimized
- Segment analysis is limited for interpretability

---

## What’s Next
- Cost-sensitive threshold tuning
- Automated threshold monitoring