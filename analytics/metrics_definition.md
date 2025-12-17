# Analytics KPI Definitions (Project 1 — PaySim)

This document defines the exported KPIs used for BI dashboards (e.g., Power BI).
All KPIs are computed from the canonical dataset: `transactions_clean.parquet`.

## Dataset grain
- One row = one transaction event.

## Fields used
- `step`: integer time step in the PaySim simulation.
- `type`: transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN).
- `amount`: transaction amount (numeric, >= 0).
- `isfraud`: fraud label (0/1).

---

## Export 1 — `kpi_daily.csv` (per step)
Grain: one row per `step`.

### Metrics
- `transactions`  
  Count of transactions at the step.  
  Formula: `COUNT(*)`

- `fraud_transactions`  
  Number of fraud transactions at the step.  
  Formula: `SUM(isfraud)`

- `fraud_rate`  
  Fraud transactions divided by total transactions at the step.  
  Formula: `AVG(isfraud)` (since isfraud is 0/1)

- `total_amount`  
  Sum of transaction amount at the step.  
  Formula: `SUM(amount)`

- `avg_amount`  
  Average transaction amount at the step.  
  Formula: `AVG(amount)`

---

## Export 2 — `kpi_by_type.csv` (by transaction type)
Grain: one row per `type`.

### Metrics
- `transactions` = `COUNT(*)`
- `fraud_transactions` = `SUM(isfraud)`
- `fraud_rate` = `AVG(isfraud)`
- `total_amount` = `SUM(amount)`
- `avg_amount` = `AVG(amount)`

---

## Notes for BI
- Fraud rate is exported as a numeric fraction (e.g., 0.0123). Format as percentage in Power BI.
- Use `step` as the time axis. If you later create calendar timestamps, add it as a separate enrichment step.