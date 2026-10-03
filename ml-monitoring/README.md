# DDM501.22 ? Wine Model Monitoring

H?c vi?n: **L? Thanh Ph??ng Nam ? 25MS23308**.

Demo tri?n khai model ph?n lo?i gi?ng r??u v?i FastAPI, MLflow,
PostgreSQL, MinIO, Evidently, Prometheus v? Grafana.
Project ???c ph?t tri?n v? ?i?u ch?nh t? source monitoring ???c cung c?p
trong m?n h?c; d? li?u, schema, metric v? quy tr?nh ch?y ?? ???c ??ng b?.

## 1. B?i to?n v? ki?n tr?c

D? li?u s? d?ng `sklearn.datasets.load_wine`: 178 m?u, 13 ??c tr?ng,
3 l?p gi?ng r??u. ??y l? b?i to?n ph?n lo?i gi?ng r??u, kh?ng ph?i
ch?m ?i?m ch?t l??ng r??u.

- Training: Random Forest, seed 42, chia train/test c? stratification.
- MLflow l?u parameters, metrics, model, signature v? reference artifacts.
- PostgreSQL l?u metadata MLflow; MinIO l?u artifacts qua S3.
- FastAPI t?i model t? Registry, ki?m tra input v? export metrics.
- Simulator g?i prediction t?i API v? capture d? li?u v?o Evidently.
- Evidently so s?nh production window v?i reference training.
- Prometheus scrape API/Evidently, ??nh gi? alert; Grafana hi?n th? dashboard.

T?n Registry `wine_quality_model` ???c gi? ?? t??ng th?ch c?u h?nh source.
Prediction tr? class ID `0`, `1` ho?c `2`.

## 2. Y?u c?u

- Docker Desktop ?ang ch?y Linux containers.
- Docker Compose h? tr? `--wait`.
- Git v? Windows PowerShell.
- Internet khi build l?n ??u.

Kh?ng c?n c?i Python tr?n host ?? ch?y project.
C?c th?ng tin ??ng nh?p trong Compose d?ng cho demo tr?n m?y c? nh?n.

## 3. Ch?y t? GitHub tr?n m?y m?i

```powershell
git clone https://github.com/NamBobby/bonus2_assignment.git
Set-Location ".\bonus2_assignment\ml-monitoring"

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1

```

## 4. ??a ch? truy c?p

| Service | URL | ??ng nh?p |
|---|---|---|
| API docs | http://localhost:28000/docs | ? |
| API health | http://localhost:28000/health | ? |
| Model schema | http://localhost:28000/model/info | ? |
| Evidently docs | http://localhost:28001/docs | ? |
| Evidently reports | http://localhost:28001/reports | ? |
| MLflow | http://localhost:25000 | ? |
| MinIO Console | http://localhost:29001 | minio / minio123 |
| Prometheus | http://localhost:29090 | ? |
| Grafana | http://localhost:23000 | admin / admin |

Grafana provision hai dashboard t? file:
`ml-model-monitoring` v? `evidently-drift`.

## 5. Ch?y th? nghi?m

Ch?y t? th? m?c `ml-monitoring`.

Normal traffic:

```powershell
docker compose run --rm --no-deps -T `
    -v "${PWD}:/work" -w /work `
    api python simulations/run_simulation.py `
    --scenario normal --requests 400 --rps 20 `
    --reset --analyze --window 400 `
    --api-url http://api:8000 `
    --evidently-url http://evidently:8001
```

Severe drift:

```powershell
docker compose run --rm --no-deps -T `
    -v "${PWD}:/work" -w /work `
    api python simulations/run_simulation.py `
    --scenario severe_drift --requests 400 --rps 20 `
    --reset --analyze --window 400 `
    --api-url http://api:8000 `
    --evidently-url http://evidently:8001
```

`--reset` x?a production window tr??c m?i profile.
Simulator d?ng v?i exit code kh?c 0 n?u prediction, capture ho?c analysis l?i.

Normal l?y m?u c? ho?n l?i t? reference training.
Severe drift d?ch chuy?n 6 feature c? ??nh th?m 3 ?? l?ch chu?n.
??y l? ki?m th? monitoring c? ki?m so?t, kh?ng ch?ng minh hi?u qu?
tr?n d? li?u production ??c l?p.

## 6. Drift v? k?t qu? ?? ki?m ch?ng

`evidently_drift_score` l? s? feature b? drift chia cho 13, kh?ng ph?i PSI.
Ng??ng dataset drift m?c ??nh l? 0.30.
Ph?n t?ch y?u c?u ?t nh?t 100 m?u production.
Reference g?m 142 m?u training; 36 m?u test d?ng ??nh gi? model.

| Profile | Prediction/capture th?nh c?ng | Feature drift | Drift score |
|---|---:|---:|---:|
| normal | 400/400 | 0/13 | 0 |
| severe_drift | 400/400 | 6/13 | 0.461538 |

S?u feature drift kh?p c?c feature ???c d?ch chuy?n:
`alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`,
`total_phenols`.

Model version 1 ??t accuracy v? weighted F1 b?ng 1.0 tr?n 36 m?u test.
K?t qu? n?y kh?ng ?? ?? k?t lu?n model ph? h?p production th?c t?.

K?t qu? ???c l?u trong:
- `data/training_summary.json`
- `data/simulation_normal.json`
- `data/simulation_severe_drift.json`

## 7. Alerts v? ki?m tra

| Alert | ?i?u ki?n | Th?i gian gi? |
|---|---|---|
| ModelApiDown | API target down | 1 ph?t |
| EvidentlyDown | Evidently target down | 1 ph?t |
| HighApiErrorRate | HTTP 5xx > 5% | 5 ph?t |
| SlowPredictions | Prediction p95 > 100 ms | 5 ph?t |
| InputDriftDetected | Dataset drift ???c ph?t hi?n | 2 ph?t |
| WidespreadInputDrift | Feature drift >= 75% | 5 ph?t |

Project hi?n th? alert trong Prometheus, ch?a c?u h?nh Alertmanager
?? g?i email ho?c th?ng b?o ngo?i h? th?ng.

Drift metric ph?n ?nh l?n analysis g?n nh?t. Capture kh?ng t? ch?y analysis;
d?ng `--analyze` ?? c?p nh?t metric. Production window n?m trong RAM;
reference v? HTML reports ???c gi? trong Docker volumes.

Ki?m tra c?u h?nh:

```powershell
docker compose run --rm --no-deps -T --entrypoint promtool `
    prometheus check config /etc/prometheus/prometheus.yml
```

Ki?m tra h?nh vi alert:

```powershell
docker run --rm --entrypoint promtool `
    -v "${PWD}/config/prometheus:/rules:ro" -w /rules `
    prom/prometheus:v2.51.2 test rules alert_tests.yml
```

Ki?m tra service:

```powershell
docker compose ps
docker compose logs --tail 80 api evidently mlflow
Invoke-RestMethod "http://localhost:28000/health"
Invoke-RestMethod "http://localhost:28001/health"
```

## 8. N?p l?i reference v? d?ng h? th?ng

N?u thi?u reference ho?c v?a train l?i:

```powershell
docker compose run --rm --no-deps -T `
    -e EVIDENTLY_URL=http://evidently:8001 `
    -v "${PWD}:/work" -w /work `
    api python scripts/upload_reference.py

Invoke-RestMethod "http://localhost:28000/model/reload" -Method Post
```

Reset production window tr??c th? nghi?m m?i.

D?ng h? th?ng, gi? d? li?u volumes:

```powershell
docker compose down
```

`docker compose down -v` x?a volumes, bao g?m Registry, artifacts,
reference, reports v? l?ch s? monitoring; sau ?? c?n ch?y setup l?i.

C?c script simulator/shell kh?c t? source ban ??u ch? ?? tham kh?o.
Quy tr?nh ???c ki?m ch?ng s? d?ng `scripts/setup.ps1` v?
`simulations/run_simulation.py` nh? h??ng d?n tr?n.
