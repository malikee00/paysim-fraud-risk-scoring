# Data Split Report (Temporal)

## Contract
- split.method: temporal
- train_max_step (T): 400
- test_min_step (T+1): 401

## Dataset Overview
- Total rows: 6,362,620
- Step range: 1 → 743

## Split Summary
| Split | Rows | Step Range | Fraud Rate |
|---|---:|---|---:|
| Train | 5,787,030 | 1 → 400 | N/A |
| Test | 575,590 | 401 → 743 | N/A |

## Sanity Checks
 No overlap: max(train.step)=400 < min(test.step)=401
 Temporal split (anti-leakage): train uses past steps only
