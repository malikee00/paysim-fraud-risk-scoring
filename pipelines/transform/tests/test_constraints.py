import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed/transactions_clean.parquet")

VALID_TYPES = {
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
    "DEBIT",
    "CASH_IN",
}


def test_amount_non_negative():
    df = pd.read_parquet(PROCESSED)
    assert (df["amount"] >= 0).all(), "Found negative amount values"


def test_step_non_null():
    df = pd.read_parquet(PROCESSED)
    assert df["step"].notna().all(), "Null values found in step column"


def test_type_valid_categories():
    df = pd.read_parquet(PROCESSED)
    invalid = set(df["type"].dropna().unique()) - VALID_TYPES
    assert not invalid, f"Invalid transaction types found: {invalid}"


def test_isfraud_binary():
    df = pd.read_parquet(PROCESSED)
    invalid = set(df["isfraud"].dropna().unique()) - {0, 1}
    assert not invalid, f"isFraud contains non-binary values: {invalid}"