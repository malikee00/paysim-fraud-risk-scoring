# Data Understanding & Policy

## Context
This project uses **PaySim**, a simulated mobile money transaction dataset designed to reflect real-world transaction behavior, including class imbalance and temporal dependency.

Although synthetic, the dataset is suitable for demonstrating **production-style ML system design**.

---

## Key Decisions
- Use **event-level transaction data** (not user profiles)
- Apply **temporal split** to avoid future data leakage
- Commit only **small sample data** to the repository
- Full dataset is fetched reproducibly via scripts

---

## Evidence / Artefacts
- Data fetch logic:  
  `pipelines/ingest/fetch_data.py`
- Schema validation:  
  `pipelines/ingest/validate_schema.py`
- Sample dataset:  
  `data/raw/sample.csv`

---

## Trade-offs / Assumptions
- Dataset is simulated and may not reflect adaptive attackers
- Time is represented as discrete steps, not timestamps

---

## What’s Next
- Add drift checks against inference-time data
- Introduce dataset versioning
