from __future__ import annotations

import time
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
)
from app.api.deps import get_artifacts
from ml.inference.predict import predict_single

def log_event(payload: dict):
    print(payload)

app = FastAPI(
    title="PaySim Fraud Risk API",
    version="1.0.0",
)

# =========================
# CORS 
# =========================
DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
origins_env = os.getenv("CORS_ORIGINS", DEFAULT_ORIGINS)
allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

if "*" not in allow_origins:
    allow_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,       
    allow_credentials=True,
    allow_methods=["*"],               
    allow_headers=["*"],               
)

# =========================
# Health check
# =========================
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    artifacts = get_artifacts()
    return HealthResponse(
        status="ok",
        model_version=artifacts.model_version,
    )

# =========================
# Prediction endpoint
# =========================
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    start_time = time.perf_counter()
    artifacts = get_artifacts()

    try:
        # ── STEP 1: validate input
        if req.features is None:
            raise HTTPException(
                status_code=400,
                detail="Raw transaction input not supported yet. Use `features` payload.",
            )

        # ── STEP 3 & 4: inference + thresholds
        # result sekarang mengandung 'thresholds_version' dari perubahan predict.py sebelumnya
        result = predict_single(req.features, artifacts)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # ── STEP 5: logging
        log_event(
            {
                "event": "predict",
                "status": "success",
                "latency_ms": round(latency_ms, 2),
                "model_version": artifacts.model_version,
                "thresholds_version": result.get("thresholds_version", "unknown"),
                "bucket": result["bucket"],
                "action": result["action"],
            }
        )

        return PredictResponse(
            risk_score=result["risk_score"],
            risk_bucket=result["bucket"],
            recommended_action=result["action"],
            model_version=artifacts.model_version,
            # Ambil langsung dari hasil fungsi predict_single
            thresholds_version=result.get("thresholds_version", "unknown"), 
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        log_event(
            {
                "event": "predict",
                "status": "error",
                "latency_ms": round(latency_ms, 2),
                "model_version": artifacts.model_version,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")