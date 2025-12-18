# Evaluation Report

## A) Global Metrics
- **PR-AUC (primary)**: **0.899141**

### Precision/Recall @ Candidate Thresholds
| Threshold | Precision | Recall | Pred_Pos_Rate |
|---:|---:|---:|---:|
| 0.05 | 0.217542 | 0.988490 | 0.029493 |
| 0.10 | 0.345746 | 0.934422 | 0.017542 |
| 0.15 | 0.506602 | 0.893469 | 0.011447 |
| 0.20 | 0.636706 | 0.860814 | 0.008775 |
| 0.25 | 0.795175 | 0.829229 | 0.006769 |
| 0.30 | 0.859669 | 0.819861 | 0.006190 |
| 0.35 | 0.871429 | 0.816381 | 0.006081 |
| 0.40 | 0.894612 | 0.808887 | 0.005869 |
| 0.45 | 0.934906 | 0.795771 | 0.005525 |
| 0.50 | 0.946543 | 0.791488 | 0.005427 |
| 0.55 | 0.954250 | 0.787206 | 0.005355 |
| 0.60 | 0.961488 | 0.781852 | 0.005278 |
| 0.65 | 0.978861 | 0.768469 | 0.005096 |
| 0.70 | 0.987754 | 0.755621 | 0.004965 |
| 0.75 | 0.991166 | 0.750803 | 0.004917 |
| 0.80 | 0.992190 | 0.748126 | 0.004894 |
| 0.85 | 0.995337 | 0.742773 | 0.004844 |
| 0.90 | 0.998168 | 0.729390 | 0.004743 |
| 0.95 | 0.998891 | 0.723233 | 0.004700 |

## B) Threshold Selection (T1/T2)
- **T1 (approve→review)**: **0.50**
- **T2 (review→block)**: **0.85**

Decision mapping:
- score < T1 → **APPROVE (Low risk)**
- T1 ≤ score < T2 → **REVIEW (Medium risk)**
- score ≥ T2 → **BLOCK (High risk)**

## C) Bucket Performance
| Bucket | Volume | Volume Rate | Key Metric |
|---|---:|---:|---|
| Low | 572466 | 0.994573 | fp_rate_flagged_at_T1=0.000292 |
| Medium | 336 | 0.000584 | precision_in_review=0.541667 |
| High | 2788 | 0.004844 | fraud_recall_in_high=0.742773 |

### Bucket Rates
- approve_rate: 0.994573
- review_rate: 0.000584
- block_rate: 0.004844


## Error Analysis
Fokus: (1) **False Negative yang lolos (APPROVE)** dan (2) **False Positive yang keblok (BLOCK)**.

### False Negatives — Fraud APPROVED (worst-case)
- Showing top 5 cases (prioritize high amount / high score)

|   _score |   _bucket |
|---------:|----------:|
| 0.492985 |         0 |
| 0.491587 |         0 |
| 0.489648 |         0 |
| 0.485595 |         0 |
| 0.47642  |         0 |

### False Positives — Legit BLOCKED (most painful)
- Showing top 5 cases (prioritize high amount / high score)

|   _score |   _bucket |
|---------:|----------:|
| 0.994593 |         2 |
| 0.982606 |         2 |
| 0.954632 |         2 |
| 0.939497 |         2 |
| 0.939156 |         2 |

### Hypotheses & Next Actions (3–5 bullets)
- FN approve cenderung terjadi pada pola velocity/sequence yang belum tertangkap (butuh fitur tx_count_recent / amount_sum_recent).
- FP block kemungkinan pada transaksi legitimate beramount tinggi yang mirip fraud; butuh fitur baseline perilaku per user (typical_amount / zscore).
- Pertimbangkan score calibration (Platt/Isotonic) agar confidence lebih stabil sebelum menetapkan T2 sangat tinggi.
- Tambahkan evaluasi per segmen (type, amount bucket, user activity) untuk menemukan blind spot spesifik segmen.

_(Optional) Full cases exported to `ml/reports/error_cases.csv`_

## D) Segment-level Evaluation (Top worst)
### By type
_Skipped (segment column not available)_

### By amount bucket
_Skipped (segment column not available)_

### By user activity proxy
_Skipped (segment column not available)_

