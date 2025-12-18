# Train Summary — Baseline V1 (logreg)

## Data Contract
- Split: temporal (anti-leakage)
- Train steps: ≤ 400
- Test steps: ≥ 401
- Dataset: `data/processed_local/features_v1_full.parquet`

## Data Size
- Total rows: 6,362,620
- Train rows (full split): 5,787,030
- Train rows used (cap): **1,000,000**
- Test rows: 575,590

## Model
- Model type: logreg
- Model name: logreg_v1
- Params: `{"max_iter": 200, "solver": "lbfgs", "class_weight": "balanced", "n_jobs": -1, "random_state": 42}`

## Metrics
- **PR-AUC (Average Precision)**: **0.733998**

### Precision/Recall @ Thresholds
| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.10 | 0.046282 | 0.996788 |
| 0.20 | 0.056597 | 0.996788 |
| 0.30 | 0.064998 | 0.996788 |
| 0.50 | 0.086607 | 0.996788 |
| 0.70 | 0.145431 | 0.941916 |

### Confusion Matrix (selected threshold)
- Selected threshold: 0.50

| | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 532,579 | 39,275 |
| True 1 | 12 | 3,724 |

## Engineering
- Training runtime: 22.61 sec
- Model size: 1.39 KB
