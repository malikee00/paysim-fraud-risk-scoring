from __future__ import annotations

from typing import Any, Dict, Literal, Optional, List

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Allowed PaySim transaction types
TxnType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


class RawTransactionIn(BaseModel):
    step: int = Field(..., ge=0, description="PaySim time step (0..)")
    type: TxnType = Field(..., description="Transaction type")
    amount: float = Field(..., ge=0, description="Transaction amount")

    oldbalanceOrg: float = Field(..., ge=0, description="Origin old balance")
    newbalanceOrig: float = Field(..., ge=0, description="Origin new balance")
    oldbalanceDest: float = Field(..., ge=0, description="Dest old balance")
    newbalanceDest: float = Field(..., ge=0, description="Dest new balance")

    nameOrig: Optional[str] = Field(default=None, description="Origin account ID (optional)")
    nameDest: Optional[str] = Field(default=None, description="Destination account ID (optional)")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: Optional[RawTransactionIn] = None
    features: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def check_exactly_one(self):
        has_raw = self.raw is not None
        has_features = self.features is not None

        if has_raw == has_features:  # both True OR both False
            raise ValueError("Provide exactly one of: `raw` or `features`.")
        return self


RiskBucket = Literal["low", "medium", "high"]
RecommendedAction = Literal["approve", "review", "block"]


class PredictResponse(BaseModel):
    risk_score: float = Field(..., ge=0, le=1, description="Fraud risk score (0..1)")
    risk_bucket: RiskBucket
    recommended_action: RecommendedAction

    model_version: Optional[str] = None
    thresholds_version: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str

class BatchPredictResponse(BaseModel):
    n_rows: int
    n_success: int
    n_failed: int
    results: List[Dict[str, Any]]