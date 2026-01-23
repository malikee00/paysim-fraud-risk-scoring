# Real-Time Transaction Risk Scoring System

> **TL;DR** > An end-to-end machine learning system that scores transaction risk in real-time, 
> translates predictions into operational decisions (approve / review / block), 
> and demonstrates production-ready ML engineering from data pipelines to API serving.

---

## 🚀 Live Demo
![Demo Paysim](docs/demo_paysim.gif)

Experience the fully deployed application:

- **Frontend (Web App)**: [https://paysim-fraud-risk-scoring.vercel.app](https://paysim-fraud-risk-scoring.vercel.app)
- **API Documentation**: [https://paysim-fraud-risk-scoring.onrender.com/docs](https://paysim-fraud-risk-scoring.onrender.com/docs)
- **API Base URL**: `https://paysim-fraud-risk-scoring.onrender.com`

---

## 🔗 API Endpoints

The system supports direct integration via REST API:

- `GET /health`: Check backend server health status.
- `POST /predict`: Real-time prediction for single transaction input.
- `POST /predict_batch`: CSV upload for bulk transaction scoring (integrated into Web App).

> **Note**: The API supports both **raw transaction input** and **feature-level input**, ensuring consistency between training and inference environments.

---
## 🛠️ Tech Stack

- **Machine Learning**: Scikit-Learn, Joblib (Model Serialization)
- **Backend**: FastAPI (Python), Uvicorn
- **Frontend**: Next.js (React), Tailwind CSS
- **Deployment**: Render (API), Vercel (Web App)
- **Data Handling**: Pandas, NumPy
---

## 🏗️ System Architecture

![architecture](docs/architecture.png)

**High-level Workflow**:
1. **Data Ingestion**: Transaction data validation and ingestion.
2. **Feature Engineering**: Canonical transformation (V1 → V2) and automated feature building.
3. **Model Development**: Training and evaluation (from baseline to optimized models).
4. **Model Registry**: Management of model versions and operational thresholds.
5. **Real-time Serving**: Low-latency inference powered by FastAPI.
6. **Monitoring**: Basic latency tracking and data drift detection.

---

## 💻 Quickstart (Local Development)

To run this project locally, follow these steps:

```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Execute data pipeline & build features
python scripts/run_etl.py
python scripts/build_features.py

# 3. Train & evaluate the model
python scripts/train.py
python scripts/evaluate.py

# 4. Start Local API
python main_render.py