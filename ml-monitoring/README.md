# DDM501.22 — Wine Model Monitoring

Học viên: **Lê Thanh Phương Nam — 25MS23308**.

Demo triển khai model phân loại giống rượu với FastAPI, MLflow,
PostgreSQL, MinIO, Evidently, Prometheus và Grafana.
Project được phát triển và điều chỉnh từ source monitoring được cung cấp
trong môn học; dữ liệu, schema, metric và quy trình chạy đã được đồng bộ.

## 1. Bài toán và kiến trúc

Dữ liệu sử dụng `sklearn.datasets.load_wine`: 178 mẫu, 13 đặc trưng,
3 lớp giống rượu. Đây là bài toán phân loại giống rượu, không phải
chấm điểm chất lượng rượu.

- Training: Random Forest, seed 42, chia train/test có stratification.
- MLflow lưu parameters, metrics, model, signature và reference artifacts.
- PostgreSQL lưu metadata MLflow; MinIO lưu artifacts qua S3.
- FastAPI tải model từ Registry, kiểm tra input và export metrics.
- Simulator gửi prediction tới API và capture dữ liệu vào Evidently.
- Evidently so sánh production window với reference training.
- Prometheus scrape API/Evidently, đánh giá alert; Grafana hiển thị dashboard.

Tên Registry `wine_quality_model` được giữ để tương thích cấu hình source.
Prediction trả class ID `0`, `1` hoặc `2`.

## 2. Yêu cầu

- Docker Desktop đang chạy Linux containers.
- Docker Compose hỗ trợ `--wait`.
- Git và Windows PowerShell.
- Internet khi build lần đầu.

Không cần cài Python trên host để chạy project.
Các thông tin đăng nhập trong Compose dùng cho demo trên máy cá nhân.

## 3. Chạy từ GitHub trên máy mới

```powershell
git clone https://github.com/NamBobby/bonus2_assignment.git
Set-Location ".\bonus2_assignment\ml-monitoring"

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

## 4. Địa chỉ truy cập

| Service | URL | Đăng nhập |
|---|---|---|
| API docs | http://localhost:28000/docs | — |
| API health | http://localhost:28000/health | — |
| Model schema | http://localhost:28000/model/info | — |
| Evidently docs | http://localhost:28001/docs | — |
| Evidently reports | http://localhost:28001/reports | — |
| MLflow | http://localhost:25000 | — |
| MinIO Console | http://localhost:29001 | minio / minio123 |
| Prometheus | http://localhost:29090 | — |
| Grafana | http://localhost:23000 | admin / admin |

Grafana provision hai dashboard từ file:
`ml-model-monitoring` và `evidently-drift`.

## 5. Chạy thử nghiệm

Chạy từ thư mục `ml-monitoring`.

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

`--reset` xóa production window trước mỗi profile.
Simulator dừng với exit code khác 0 nếu prediction, capture hoặc analysis lỗi.

Normal lấy mẫu có hoàn lại từ reference training.
Severe drift dịch chuyển 6 feature cố định thêm 3 độ lệch chuẩn.
Đây là kiểm thử monitoring có kiểm soát, không chứng minh hiệu quả
trên dữ liệu production độc lập.

## 6. Drift và kết quả đã kiểm chứng

`evidently_drift_score` là số feature bị drift chia cho 13, không phải PSI.
Ngưỡng dataset drift mặc định là 0.30.
Phân tích yêu cầu ít nhất 100 mẫu production.
Reference gồm 142 mẫu training; 36 mẫu test dùng đánh giá model.

| Profile | Prediction/capture thành công | Feature drift | Drift score |
|---|---:|---:|---:|
| normal | 400/400 | 0/13 | 0 |
| severe_drift | 400/400 | 6/13 | 0.461538 |

Sáu feature drift khớp các feature được dịch chuyển:
`alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`,
`total_phenols`.

Model version 1 đạt accuracy và weighted F1 bằng 1.0 trên 36 mẫu test.
Kết quả này không đủ để kết luận model phù hợp production thực tế.

Kết quả được lưu trong:

- `data/training_summary.json`
- `data/simulation_normal.json`
- `data/simulation_severe_drift.json`

## 7. Alerts và kiểm tra

| Alert | Điều kiện | Thời gian giữ |
|---|---|---|
| ModelApiDown | API target down | 1 phút |
| EvidentlyDown | Evidently target down | 1 phút |
| HighApiErrorRate | HTTP 5xx > 5% | 5 phút |
| SlowPredictions | Prediction p95 > 100 ms | 5 phút |
| InputDriftDetected | Dataset drift được phát hiện | 2 phút |
| WidespreadInputDrift | Feature drift >= 75% | 5 phút |

Project hiển thị alert trong Prometheus, chưa cấu hình Alertmanager
để gửi email hoặc thông báo ngoài hệ thống.

Drift metric phản ánh lần analysis gần nhất. Capture không tự chạy analysis;
dùng `--analyze` để cập nhật metric. Production window nằm trong RAM;
reference và HTML reports được giữ trong Docker volumes.

Kiểm tra cấu hình:

```powershell
docker compose run --rm --no-deps -T --entrypoint promtool `
    prometheus check config /etc/prometheus/prometheus.yml
```

Kiểm tra hành vi alert:

```powershell
docker run --rm --entrypoint promtool `
    -v "${PWD}/config/prometheus:/rules:ro" -w /rules `
    prom/prometheus:v2.51.2 test rules alert_tests.yml
```

Kiểm tra service:

```powershell
docker compose ps
docker compose logs --tail 80 api evidently mlflow
Invoke-RestMethod "http://localhost:28000/health"
Invoke-RestMethod "http://localhost:28001/health"
```

## 8. Nạp lại reference và dừng hệ thống

Nếu thiếu reference hoặc vừa train lại:

```powershell
docker compose run --rm --no-deps -T `
    -e EVIDENTLY_URL=http://evidently:8001 `
    -v "${PWD}:/work" -w /work `
    api python scripts/upload_reference.py

Invoke-RestMethod "http://localhost:28000/model/reload" -Method Post
```

Reset production window trước thử nghiệm mới.

Dừng hệ thống, giữ dữ liệu volumes:

```powershell
docker compose down
```

`docker compose down -v` xóa volumes, bao gồm Registry, artifacts,
reference, reports và lịch sử monitoring; sau đó cần chạy setup lại.

Các script simulator/shell khác từ source ban đầu chỉ để tham khảo.
Quy trình được kiểm chứng sử dụng `scripts/setup.ps1` và
`simulations/run_simulation.py` như hướng dẫn trên.
