Write-Host "Running drift check..."

python scripts/check_drift.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Drift check failed"
    exit 1
}

Write-Host "Drift check completed"
