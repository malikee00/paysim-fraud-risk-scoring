Write-Host "Running latency benchmark..."

python scripts/benchmark_latency.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Latency benchmark failed"
    exit 1
}

Write-Host "Latency benchmark completed"
