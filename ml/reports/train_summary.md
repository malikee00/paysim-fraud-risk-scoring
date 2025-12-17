# Train Summary — Baseline V1 (LogReg)

## Data Contract
- Split: temporal (anti-leakage)
- Train steps: ≤ 400
- Test steps: ≥ 401
- Dataset: `data/processed_local/features_v1_full.parquet`

## Data Size
- Total rows: 6,362,620
- Train rows (full split): 5,787,030
- Train rows used (cap): **500,000**
- Test rows: 575,590

## Model
- Model: Logistic Regression
- class_weight: balanced
- Params: `{"max_iter": 200, "solver": "lbfgs", "class_weight": "balanced", "n_jobs": -1, "random_state": 42}`

## Metrics
- **PR-AUC (Average Precision)**: **0.816345**

### Precision/Recall @ Thresholds
| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.10 | 0.045778 | 0.996788 |
| 0.20 | 0.056884 | 0.996788 |
| 0.30 | 0.066545 | 0.996788 |
| 0.50 | 0.089114 | 0.996788 |
| 0.70 | 0.194147 | 0.912741 |

### Confusion Matrix (selected threshold)
- Selected threshold: 0.50

| | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 533,789 | 38,065 |
| True 1 | 12 | 3,724 |

## Engineering
- Training runtime: 10.99 sec
- Model size: 1.39 KB
