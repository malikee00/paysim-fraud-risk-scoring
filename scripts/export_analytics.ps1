param(
  [string]$InputParquet = "data/processed/transactions_clean.parquet",
  [string]$OutDir = "analytics/exports"
)

Write-Host "=== ANALYTICS EXPORT ==="
Write-Host "Input : $InputParquet"
Write-Host "OutDir: $OutDir"

python analytics/export_analytics.py --input "$InputParquet" --out_dir "$OutDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n Analytics export done."
