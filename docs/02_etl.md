# ETL & Data Quality

## Context
Reliable machine learning systems require clean, validated, and reproducible data pipelines.

This project implements a lightweight **ETL pipeline with explicit data quality checks**, inspired by production best practices.

---

## Key Decisions
- Separate pipeline stages:
  - ingest
  - validate
  - transform
- Enforce schema and value constraints before feature building
- Produce a canonical transaction table reusable for ML and BI

---

## Evidence / Artefacts
- Transform pipeline:  
  `pipelines/transform/transform.py`
- Data quality tests:  
  - `pipelines/transform/tests/test_row_count.py`  
  - `pipelines/transform/tests/test_constraints.py`
- ETL runner script:  
  `scripts/run_etl.ps1`

---

## Trade-offs / Assumptions
- ETL uses pandas instead of distributed processing
- Data quality rules are heuristic, not domain-certified

---

## What’s Next
- Add automated ETL checks in CI
- Extend quality checks with statistical profiling