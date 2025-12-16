# scripts/validate_data.ps1
$InputFile = $args[0]
if (-not $InputFile) {
    $InputFile = "data/raw/sample_raw.csv"
}

Write-Host "Running schema validation on: $InputFile"
python pipelines/ingest/validate_schema.py --input "$InputFile" --strict