# Train Summary — Improved V2 (hgb)

## Data Contract
- Split: temporal (anti-leakage)
- Train steps: ≤ 400
- Test steps: ≥ 401
- Dataset: `data/processed_local/features_v2_full.parquet`

## Data Size
- Total rows: 6,362,620
- Train rows (full split): 5,787,030
- Train rows used (cap): **200,000**
- Test rows: 575,590

## Model
- Model type: hgb
- Model name: hgb_v2_dest_only
- Params: `{"max_depth": 6, "learning_rate": 0.1, "max_iter": 200, "random_state": 42}`

## Metrics
- **PR-AUC (Average Precision)**: **0.732431**

### Precision/Recall @ Thresholds
| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.10 | 0.786310 | 0.707173 |
| 0.20 | 0.793155 | 0.707173 |
| 0.30 | 0.819400 | 0.687366 |
| 0.50 | 0.980717 | 0.585385 |
| 0.70 | 0.977953 | 0.498662 |

### Confusion Matrix (selected threshold)
- Selected threshold: 0.50

| | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 571,811 | 43 |
| True 1 | 1,549 | 2,187 |

## Engineering
- Training runtime: 4.17 sec
- Model size: 63.30 KB
