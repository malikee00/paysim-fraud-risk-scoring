# Monitoring & Reliability

## Context
Even in a demo environment, basic monitoring signals are required to assess system reliability and behavior.

This project implements **lightweight monitoring** without heavy observability infrastructure.

---

## Key Decisions
- Log inference latency and prediction outcomes
- Implement offline data drift checks on key features
- Keep monitoring logic decoupled from inference logic

---

## Evidence / Artefacts
- Logging design:  
  `ops/monitoring/logging.md`
- Drift check script:  
  `scripts/check_drift.sh`
- Latency benchmark:  
  `scripts/benchmark_latency.sh`

---

## Trade-offs / Assumptions
- No real-time alerting
- Drift checks are batch-based

---

## What’s Next
- Real-time metric aggregation
- Dashboard-based monitoring and alerting