# =========================
# Build Docker Image
# =========================

$ImageName = "paysim-risk-api"
$ImageTag = "latest"
$DockerfilePath = "ops/docker/Dockerfile"

Write-Host "Building Docker image: ${ImageName}:${ImageTag}" -ForegroundColor Cyan

docker build `
  -f $DockerfilePath `
  -t "${ImageName}:${ImageTag}" `
  .

if ($LASTEXITCODE -ne 0) {
  Write-Error "Docker build failed"
  exit 1
}

Write-Host "Docker image built successfully" -ForegroundColor Green