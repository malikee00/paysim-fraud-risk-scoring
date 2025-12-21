from __future__ import annotations

import io
import pandas as pd
import os
import time
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import PredictRequest, PredictResponse, HealthResponse, BatchPredictResponse
from app.api.deps import get_artifacts

from ml.inference.predict import predict_single
from ml.inference.featurize import raw_to_features


def log_event(payload: dict) -> None:
    print(payload)

app = FastAPI(
    title="PaySim Fraud Risk API",
    version="1.0.0",
)

# =========================
# CORS (SAFE)
# =========================
# IMPORTANT:
# - If allow_credentials=True, DO NOT use "*" in allow_origins.
DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
origins_env = os.getenv("CORS_ORIGINS", DEFAULT_ORIGINS)
allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

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
    return HealthResponse(status="ok", model_version=artifacts.model_version)

@app.get("/template/raw")
def download_raw_template() -> Response:
    """
    CSV template for RAW batch upload (recommended).
    """
    cols = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "nameOrig",
        "nameDest",
    ]
    df = pd.DataFrame([{c: "" for c in cols}])
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=template_raw.csv"},
    )

@app.get("/template/features")
def download_features_template() -> Response:
    """
    CSV template for FEATURES batch upload (advanced).
    Columns follow the model feature_names_in_.
    """
    artifacts = get_artifacts()
    cols = list(artifacts.model.feature_names_in_)
    df = pd.DataFrame([{c: "" for c in cols}])
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=template_features.csv"},
    )

# =========================
# Prediction endpoint
# =========================
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    start_time = time.perf_counter()
    artifacts = get_artifacts()

    try:
        # NOTE: schemas.py already enforces exactly one of raw/features

        # ----- Mode A: RAW (from PWA form) -> featurize -> model -----
        if req.raw is not None:
            required_features = list(artifacts.model.feature_names_in_)
            engineered = raw_to_features(req.raw.model_dump(), required_features=required_features)
            result = predict_single(engineered, artifacts)

        # ----- Mode B: FEATURES (debug/internal) -> model -----
        else:
            result = predict_single(req.features, artifacts)  

        latency_ms = (time.perf_counter() - start_time) * 1000

        log_event(
            {
                "event": "predict",
                "status": "success",
                "latency_ms": round(latency_ms, 2),
                "model_version": artifacts.model_version,
                "thresholds_version": result.get("thresholds_version", "thresholds"),
                "bucket": result["bucket"],
                "action": result["action"],
            }
        )

        return PredictResponse(
            risk_score=result["risk_score"],
            risk_bucket=result["bucket"],
            recommended_action=result["action"],
            model_version=artifacts.model_version,
            thresholds_version=result.get("thresholds_version", "thresholds"),
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

@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(
    file: UploadFile = File(...),
    mode: str = "raw",  
) -> BatchPredictResponse:
    start_time = time.perf_counter()
    artifacts = get_artifacts()

    filename = (file.filename or "").lower()
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {type(e).__name__}: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty.")

    required_features = list(artifacts.model.feature_names_in_)

    results = []
    n_success = 0
    n_failed = 0

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        tx_id = row_dict.get("nameOrig") or f"UNKNOWN_{idx}"

        try:
            if mode == "raw":
                feats = raw_to_features(row_dict, required_features=required_features)
                out = predict_single(feats, artifacts)
            elif mode == "features":
                out = predict_single(row_dict, artifacts)
            else:
                raise HTTPException(status_code=400, detail="mode must be 'raw' or 'features'")

            results.append(
                {
                    "row_index": int(idx),
                    "transaction_id": str(tx_id),
                    "status": "success",
                    "risk_score": out["risk_score"],
                    "risk_bucket": out["bucket"],
                    "recommended_action": out["action"],
                }
            )
            n_success += 1

        except Exception as e:
            results.append(
                {
                    "row_index": int(idx),
                    "transaction_id": str(tx_id),
                    "status": "failed",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )
            n_failed += 1

    latency_ms = (time.perf_counter() - start_time) * 1000
    log_event(
        {
            "event": "predict_batch",
            "status": "success",
            "mode": mode,
            "latency_ms": round(latency_ms, 2),
            "model_version": artifacts.model_version,
            "n_rows": int(len(df)),
            "n_success": int(n_success),
            "n_failed": int(n_failed),
        }
    )

    return BatchPredictResponse(
        n_rows=int(len(df)),
        n_success=int(n_success),
        n_failed=int(n_failed),
        results=results,
    )

