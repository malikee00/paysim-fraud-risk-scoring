# app/api/main.py
from __future__ import annotations

import time
from fastapi import FastAPI, HTTPException, Request

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

        # ── STEP 3 & 4: inference + thresholds (inside predict_single)
        result = predict_single(req.features, artifacts)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # ── STEP 5: logging (minimal, production-mindset)
        log_event(
            {
                "event": "predict",
                "status": "success",
                "latency_ms": round(latency_ms, 2),
                "model_version": artifacts.model_version,
                "thresholds_version": "thresholds",
                "bucket": result["bucket"],
                "action": result["action"],
            }
        )

        return PredictResponse(
        risk_score=result["risk_score"],
        risk_bucket=result["bucket"],
        recommended_action=result["action"],
        model_version=artifacts.model_version,
        thresholds_version="thresholds", 
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
        raise HTTPException(status_code=500, detail="Internal server error")
