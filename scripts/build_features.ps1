param(
  [string]$InputParquet = "data/processed/transactions_clean.parquet",
  [string]$OutFeatures = "data/processed/features_v1.parquet",
  [string]$OutSummary = "data/processed/feature_summary.json"
)

Write-Host "=== FEATURE BUILD V1 ==="
Write-Host "Input : $InputParquet"
Write-Host "Output: $OutFeatures"

python pipelines/features/build_features.py `
  --input $InputParquet `
  --output $OutFeatures `
  --summary $OutSummary

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n Feature V1 build completed."
