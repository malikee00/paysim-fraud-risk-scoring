# Train Summary — Improved V2 (hgb)

## Data Contract
- Split: temporal (anti-leakage)
- Train steps: ≤ 400
- Test steps: ≥ 401
- Dataset: `data/processed_local/features_v2_full_enhance.parquet`

## Data Size
- Total rows: 6,362,620
- Train rows (full split): 5,787,030
- Train rows used (cap): **5,787,030**
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
- **PR-AUC (Average Precision)**: **0.898682**

### Precision/Recall @ Thresholds
| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.10 | 0.337684 | 0.934957 |
| 0.30 | 0.852409 | 0.819325 |
| 0.50 | 0.946492 | 0.790685 |
| 0.70 | 0.987408 | 0.755621 |
| 0.90 | 0.998168 | 0.729390 |
| 0.95 | 0.998890 | 0.722698 |

### Confusion Matrix (selected threshold)
- Selected threshold: 0.50

| | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 571,687 | 167 |
| True 1 | 782 | 2,954 |

## Engineering
- Training runtime: 420.26 sec
- Model size: 1.20 MB
