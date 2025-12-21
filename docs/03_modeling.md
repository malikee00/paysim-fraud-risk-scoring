# Modeling Strategy

## Context
Transaction fraud detection is a highly imbalanced classification problem with strong temporal characteristics.

This project adopts an **iterative modeling approach**, starting from a stable baseline and progressing to a behavior-aware improved model.

---

## Key Decisions
- Baseline model:
  - Logistic Regression with class weighting
- Improved model:
  - Same algorithm + behavioral and velocity features
- Primary evaluation metric:
  - PR-AUC (due to class imbalance)

---

## Evidence / Artefacts
- Training pipeline:  
  `ml/training/train.py`
- Feature tables:  
  - `data/processed/features_v1.parquet`  
  - `data/processed/features_v2_full_sakti.parquet`
- Model registry:  
  `ml/models/registry.md`

---

## Trade-offs / Assumptions
- Algorithm choice favors stability over peak accuracy
- Feature windows are tuned heuristically

---

## What’s Next
- Evaluate gradient boosting models
- Add feature calibration and regularization