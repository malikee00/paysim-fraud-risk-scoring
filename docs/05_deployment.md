# Serving & Deployment

## Context
To demonstrate production readiness, the trained model is exposed through a real-time inference API and consumed by a client-facing demo.

The deployment prioritizes **clarity, stability, and reproducibility** over scale.

---

## Key Decisions
- Use FastAPI for model serving
- Load model and thresholds dynamically from configuration
- Provide both:
  - API access
  - PWA demo client

---

## Evidence / Artefacts
- API service:  
  `app/api/main.py`
- Docker image definition:  
  `ops/docker/Dockerfile`
- Demo recording:  
  `docs/demo.gif`

---

## Trade-offs / Assumptions
- No authentication layer (demo scope)
- Single-instance deployment only

---

## What’s Next
- Add token-based API authentication
- Deploy lightweight monitoring dashboard