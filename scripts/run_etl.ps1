param(
  [string]$InputPath = "data/raw/sample_raw.csv",
  [string]$OutParquet = "data/processed/transactions_clean.parquet",
  [int]$NRows = 0
)

Write-Host "=== ETL RUN ==="
Write-Host "Input: $InputPath"
Write-Host "Output: $OutParquet"

# 1) Validate schema (strict)
Write-Host "`n[1/2] Validate schema..."
if ($NRows -gt 0) {
  python pipelines/ingest/validate_schema.py --input "$InputPath" --strict --nrows $NRows
} else {
  python pipelines/ingest/validate_schema.py --input "$InputPath" --strict
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 2) Transform raw -> processed parquet
Write-Host "`n[2/2] Transform..."
if ($NRows -gt 0) {
  python pipelines/transform/transform.py --input "$InputPath" --output "$OutParquet" --nrows $NRows
} else {
  python pipelines/transform/transform.py --input "$InputPath" --output "$OutParquet"
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nETL completed successfully."
