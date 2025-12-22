# Real-Time Transaction Risk Scoring System

> **TL;DR** > An end-to-end machine learning system that scores transaction risk in real time, 
> translates predictions into operational decisions (approve / review / block), 
> and demonstrates production-ready ML engineering from data pipelines to API serving.

---

## 🚀 Live Demo

Akses aplikasi yang sudah dideploy secara publik:

- **Frontend (Web App)**: [https://paysim-fraud-risk-scoring.vercel.app](https://paysim-fraud-risk-scoring.vercel.app)
- **API Documentation**: [https://paysim-fraud-risk-scoring.onrender.com/docs](https://paysim-fraud-risk-scoring.onrender.com/docs)
- **API Base URL**: `https://paysim-fraud-risk-scoring.onrender.com`

---

## 🔗 API Endpoints

Aplikasi ini mendukung integrasi langsung melalui REST API:

- `GET /health`: Mengecek status kesehatan server backend.
- `POST /predict`: Melakukan prediksi untuk transaksi tunggal.
- `POST /predict_batch`: Mengunggah file CSV untuk prediksi massal (digunakan di Web App).

> The API supports both **raw transaction input** and **feature-level input**, ensuring consistency between training and inference.

---

## 🏗️ System Architecture

![architecture](docs/architecture.png)

**High-level flow**:
1. Transaction data ingestion & validation
2. Canonical transformation & feature engineering (V1 → V2)
3. Model training & evaluation (baseline → improved)
4. Model registry & threshold configuration
5. Real-time inference via FastAPI
6. Lightweight monitoring (latency & drift)

---

## 💻 Quickstart (Local Development)

Jika ingin menjalankan proyek ini di mesin lokal:

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

# 4. Start API (Local)
python main_render.py