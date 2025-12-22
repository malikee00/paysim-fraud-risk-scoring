# Real-Time Transaction Risk Scoring System

> **TL;DR**  
> An end-to-end machine learning system that scores transaction risk in real time,
> translates predictions into operational decisions (approve / review / block),
> and demonstrates production-ready ML engineering from data pipelines to API serving.

---

## 🔗 Demo

- **API**: `http://127.0.0.1:8000` (local)
- **Endpoints**:
  - `GET /health`
  - `POST /predict`
- **Interactive Demo**: see `docs/demo.gif`

> The API supports both **raw transaction input** and **feature-level input**, ensuring
> consistency between training and inference.

---

## System Architecture

![architecture](docs/architecture.png)

**High-level flow**:

1. Transaction data ingestion & validation  
2. Canonical transformation & feature engineering (V1 → V2)  
3. Model training & evaluation (baseline → improved)  
4. Model registry & threshold configuration  
5. Real-time inference via FastAPI  
6. Lightweight monitoring (latency & drift)

---

## Quickstart (Local)

```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Run data pipeline & feature build
python scripts/run_etl.py
python scripts/build_features.py

# 3. Train & evaluate model
python scripts/train.py
python scripts/evaluate.py

# 4. Start API
python scripts/serve.py

# 5. Smoke test
python scripts/smoke_test_api.py
