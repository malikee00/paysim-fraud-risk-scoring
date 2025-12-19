from __future__ import annotations

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator


TxnType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


class RawTransactionIn(BaseModel):
    # core PaySim fields
    step: int = Field(..., ge=0)
    type: TxnType
    amount: float = Field(..., ge=0)

    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)

    # optional IDs for future behavioral lookup
    nameOrig: Optional[str] = None
    nameDest: Optional[str] = None


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: Optional[RawTransactionIn] = None
    features: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def check_exactly_one(self):
        if (self.raw is None and self.features is None) or (self.raw is not None and self.features is not None):
            raise ValueError("Provide exactly one of: `raw` or `features`.")
        return self


class PredictResponse(BaseModel):
    risk_score: float
    risk_bucket: Literal["low", "medium", "high"]
    recommended_action: Literal["approve", "review", "block"]
    model_version: Optional[str] = None
    thresholds_version: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str
