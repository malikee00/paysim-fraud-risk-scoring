# Train Summary — Improved V2 (hgb)

## Data Contract
- Split: temporal (anti-leakage)
- Train steps: ≤ 400
- Test steps: ≥ 401
- Dataset: `data/processed_local/features_v2_full.parquet`

## Data Size
- Total rows: 6,362,620
- Train rows (full split): 5,787,030
- Train rows used (cap): **1,000,000**
- Test rows: 575,590

## Model
- Model type: hgb
- Model name: hgb_v2_improved
- Params: `{"max_depth": 12, "learning_rate": 0.05, "max_iter": 400, "min_samples_leaf": 40, "random_state": 42}`

## Metrics
- **PR-AUC (Average Precision)**: **0.848072**

### Precision/Recall @ Thresholds
| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.10 | 0.498285 | 0.816381 |
| 0.30 | 0.902897 | 0.759101 |
| 0.50 | 0.977933 | 0.747323 |
| 0.70 | 0.988873 | 0.737420 |
| 0.90 | 0.998122 | 0.711188 |
| 0.95 | 0.999235 | 0.699679 |

### Confusion Matrix (selected threshold)
- Selected threshold: 0.50

| | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 571,791 | 63 |
| True 1 | 944 | 2,792 |

## Engineering
- Training runtime: 25.80 sec
- Model size: 422.56 KB
