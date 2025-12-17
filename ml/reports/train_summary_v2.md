# Train Summary — Improved V2 (hgb)

## Data Contract
- Split: temporal (anti-leakage)
- Train steps: ≤ 400
- Test steps: ≥ 401
- Dataset: `data/processed_local/features_v2_full.parquet`

## Data Size
- Total rows: 6,362,620
- Train rows (full split): 5,787,030
- Train rows used (cap): **500,000**
- Test rows: 575,590

## Model
- Model type: hgb
- Model name: hgb_v2_improved
- Params: `{"max_depth": 12, "learning_rate": 0.05, "max_iter": 400, "min_samples_leaf": 40, "random_state": 42}`

## Metrics
- **PR-AUC (Average Precision)**: **0.843435**

### Precision/Recall @ Thresholds
| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.10 | 0.195821 | 0.945664 |
| 0.30 | 0.292339 | 0.887580 |
| 0.50 | 0.420339 | 0.829764 |
| 0.70 | 0.560121 | 0.795503 |
| 0.90 | 0.829516 | 0.765792 |
| 0.95 | 0.947492 | 0.758298 |

### Confusion Matrix (selected threshold)
- Selected threshold: 0.50

| | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 567,579 | 4,275 |
| True 1 | 636 | 3,100 |

## Engineering
- Training runtime: 13.62 sec
- Model size: 323.31 KB
