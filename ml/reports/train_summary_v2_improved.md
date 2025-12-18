# Train Summary — Improved V2 (hgb)

## Data Contract
- Split: temporal (anti-leakage)
- Train steps: ≤ 400
- Test steps: ≥ 401
- Dataset: `data/processed_local/features_v2_full_sakti.parquet`

## Data Size
- Total rows: 6,362,620
- Train rows (full split): 5,787,030
- Train rows used (cap): **1,000,000**
- Test rows: 575,590

## Model
- Model type: hgb
- Model name: hgb_v2_final_polish
- Params: `{"max_iter": 1000, "learning_rate": 0.02, "max_depth": 15, "min_samples_leaf": 30, "l2_regularization": 0.1, "random_state": 42}`

- **Hyperparameters**:
  - `max_iter`: 1000
  - `learning_rate`: 0.02
  - `max_depth`: 15
  - `min_samples_leaf`: 30
  - `l2_regularization`: 0.1
  - `random_state`: 42

## Metrics
- **PR-AUC (Average Precision)**: **0.879924**

### Precision/Recall @ Thresholds
| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.10 | 0.441361 | 0.885439 |
| 0.30 | 0.859068 | 0.784797 |
| 0.50 | 0.979367 | 0.762313 |
| 0.70 | 0.991462 | 0.745985 |
| 0.90 | 0.997817 | 0.733940 |
| 0.95 | 0.998495 | 0.710118 |

### Confusion Matrix (selected threshold)
- Selected threshold: 0.50

| | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 571,794 | 60 |
| True 1 | 888 | 2,848 |

## Engineering
- Training runtime: 39.25 sec
- Model size: 810.43 KB
