import pandas as pd
from pathlib import Path

RAW_SAMPLE = Path("data/raw/sample_raw.csv")
PROCESSED = Path("data/processed/transactions_clean.parquet")

MIN_RETAIN_RATIO = 0.99  

def test_row_count_sanity():
    assert RAW_SAMPLE.exists(), "Raw sample CSV not found"
    assert PROCESSED.exists(), "Processed parquet not found"

    raw_df = pd.read_csv(RAW_SAMPLE)
    proc_df = pd.read_parquet(PROCESSED)

    raw_rows = len(raw_df)
    proc_rows = len(proc_df)

    assert raw_rows > 0, "Raw dataset is empty"

    retain_ratio = proc_rows / raw_rows

    assert (
        retain_ratio >= MIN_RETAIN_RATIO
    ), f"Row count dropped too much: {retain_ratio:.2%} retained"
