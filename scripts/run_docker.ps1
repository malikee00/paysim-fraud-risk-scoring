# =========================
# Run Docker Container
# =========================

$ImageName = "paysim-risk-api"
$HostPort = 8000
$ContainerPort = 8000

Write-Host "Running Docker container on port ${HostPort}" -ForegroundColor Cyan

docker run --rm `
  -p "${HostPort}:${ContainerPort}" `
  $ImageName

if ($LASTEXITCODE -ne 0) {
  Write-Error "Docker run failed"
  exit 1
}