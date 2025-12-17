# Data Split Report (Temporal)

## Contract
- split.method: temporal
- train_max_step (T): 600
- test_min_step (T+1): 601

## Dataset Overview
- Total rows: 6,362,620
- Step range: 1 → 743

## Split Summary
| Split | Rows | Step Range | Fraud Rate |
|---|---:|---|---:|
| Train | 6,259,047 | 1 → 600 | N/A |
| Test | 103,573 | 601 → 743 | N/A |

## Sanity Checks
 No overlap: max(train.step)=600 < min(test.step)=601
 Temporal split (anti-leakage): train uses past steps only
