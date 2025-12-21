# Drift Report

Feature drift check using mean shift (proxy recent samples).

- Drift threshold: 30% relative mean change

- Train rows: 6362620

## log_amount
- Train mean: 10.840874
- Recent mean: 10.419761
- Relative change: 3.88%
- Status: **OK**

## org_delta_mismatch
- Train mean: 201092.484375
- Recent mean: 49782.077248
- Relative change: 75.24%
- Status: **POTENTIAL DRIFT**

## dest_delta_mismatch
- Train mean: 93599.070312
- Recent mean: 283960.979712
- Relative change: 203.38%
- Status: **POTENTIAL DRIFT**

## dest_txn_count_w10
- Train mean: 3.479552
- Recent mean: 1.537000
- Relative change: 55.83%
- Status: **POTENTIAL DRIFT**

## ratio_amt_to_dest_mean_w10
- Train mean: 27090835456.000000
- Recent mean: 367825889199.124084
- Relative change: 1257.75%
- Status: **POTENTIAL DRIFT**
