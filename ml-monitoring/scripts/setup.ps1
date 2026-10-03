$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Run-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) {
        throw "Docker command failed. See the output above."
    }
}

Run-Docker compose config --quiet

# Storage must be healthy before bucket initialization and training.
Run-Docker compose up -d --build --wait --wait-timeout 240 postgres minio mlflow
Run-Docker compose run --rm --no-deps -T minio-init

Run-Docker compose build api
Run-Docker compose run --rm --no-deps -T `
    -e MLFLOW_TRACKING_URI=http://mlflow:5000 `
    -e MLFLOW_S3_ENDPOINT_URL=http://minio:9000 `
    -e AWS_ACCESS_KEY_ID=minio `
    -e AWS_SECRET_ACCESS_KEY=minio123 `
    -e MODEL_NAME=wine_quality_model `
    -v "${PWD}:/work" -w /work `
    api python scripts/training.py

Run-Docker compose up -d --build --wait --wait-timeout 240 api evidently

Run-Docker compose run --rm --no-deps -T `
    -e EVIDENTLY_URL=http://evidently:8001 `
    -v "${PWD}:/work" -w /work `
    api python scripts/upload_reference.py

Invoke-RestMethod "http://localhost:28000/model/reload" -Method Post | Out-Null

Run-Docker compose run --rm --no-deps -T `
    --entrypoint promtool `
    prometheus check config /etc/prometheus/prometheus.yml

Run-Docker run --rm --entrypoint promtool `
    -v "${PWD}/config/prometheus:/rules:ro" -w /rules `
    prom/prometheus:v2.51.2 test rules alert_tests.yml

Run-Docker compose up -d prometheus grafana
Run-Docker compose ps
Write-Host "Setup completed. Grafana: http://localhost:23000"
