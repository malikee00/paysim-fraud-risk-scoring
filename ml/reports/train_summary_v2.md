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
- Model name: hgb_v2_dest_only
- Params: `{"max_depth": 6, "learning_rate": 0.1, "max_iter": 200, "random_state": 42}`

## Metrics
- **PR-AUC (Average Precision)**: **0.838000**

### Precision/Recall @ Thresholds
| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.10 | 0.138732 | 0.992773 |
| 0.20 | 0.149628 | 0.980996 |
| 0.30 | 0.162094 | 0.964668 |
| 0.50 | 0.189913 | 0.936296 |
| 0.70 | 0.273648 | 0.882762 |

### Confusion Matrix (selected threshold)
- Selected threshold: 0.50

| | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 556,933 | 14,921 |
| True 1 | 238 | 3,498 |

## Engineering
- Training runtime: 8.27 sec
- Model size: 208.28 KB
