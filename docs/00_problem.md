# Problem Framing

## Context
Digital payment platforms process large volumes of transactions, where a small fraction of fraudulent events can cause disproportionate financial and operational impact.

This project simulates a **real-time transaction risk scoring system** that converts machine learning predictions into **actionable operational decisions**, not just offline fraud detection.

The focus is on building a **deployable, decision-aware ML system**, rather than optimizing model accuracy alone.

---

## Key Decisions
- Problem framed as **event-level, real-time risk scoring**
- No user-level profiling or long-term identity modeling
- Model output designed as:
  - `risk_score` → `risk_bucket` → `recommended_action`
- System optimized for **operational decision support**, not forensic investigation

---

## Evidence / Artefacts
- Risk objective & decision design:  
  `docs/design/01_objective.md`
- API output schema:  
  `app/api/schemas.py`
- Decision thresholds configuration:  
  `ml/reports/thresholds.yaml`

---

## Trade-offs / Assumptions
- Dataset is **simulated (PaySim)**, not real financial data
- No explicit monetary cost matrix is used
- Fraud is defined strictly by the dataset label (`isFraud`)
- Fraud rings and identity resolution are out of scope

---

## What’s Next
- Introduce cost-sensitive threshold optimization
- Extend from event-level to aggregated user risk scoring