# DDM501.22 — Bonus 2: Monitoring với Prometheus và Grafana

- Học viên: Lê Thanh Phương Nam
- Mã học viên: 25MS23308
- Môn học: AI trong sản xuất — DevOps, DataOps, MLOps
- Source nền: Tutorial 04 — Prometheus and Grafana của môn học.

## Mục tiêu

Giám sát API phân loại WDBC bằng Counter, Histogram và Gauge;
kiểm chứng alert khi request lỗi, phân phối prediction thay đổi hoặc
API ngừng hoạt động; minh họa rủi ro label cardinality.

Ứng dụng là demo học tập, không dùng để đưa ra quyết định y tế.

## Yêu cầu

Docker Desktop chạy Linux containers và Docker Compose v2.
Không cần cài Python trên máy để chạy theo hướng dẫn Docker.

## Khởi chạy

Tại thư mục chứa docker-compose.yml:

```powershell
docker compose up -d --build --wait --wait-timeout 180
docker compose ps
```

Dockerfile tự train model từ data/raw/wdbc.csv trong quá trình build.
Model và môi trường serving sử dụng cùng phiên bản scikit-learn.

| Thành phần | Địa chỉ |
|---|---|
| API docs | http://localhost:18000/docs |
| Health | http://localhost:18000/health |
| Metrics | http://localhost:18000/metrics |
| Prometheus | http://localhost:19090 |
| Grafana | http://localhost:13000 |

Grafana có anonymous Viewer. Tài khoản quản trị demo: admin/admin.
Dashboard WDBC được provision từ file, datasource trỏ tới Prometheus.

## Metric

| Metric | Loại | Ý nghĩa |
|---|---|---|
| wdbc_predictions_total | Counter | Prediction thành công theo outcome |
| wdbc_errors_total | Counter | Lỗi được ghi nhận theo reason |
| wdbc_prediction_latency_seconds | Histogram | Thời gian chuẩn bị frame và chạy model |
| wdbc_model_loaded | Gauge | Model đã được load |
| wdbc_model_info | Gauge | Phiên bản model và scikit-learn qua label |
| wdbc_malignant_share | Gauge | Malignant share trong 200 prediction gần nhất |

Counter được đọc bằng rate() khi cần tốc độ hoặc tỷ lệ.
Histogram giúp quan sát p50/p95/p99 thay vì chỉ dùng trung bình.
Latency ở đây không phải toàn bộ thời gian HTTP request.

Counter có label chỉ xuất hiện sau khi label đó được sử dụng.
Metric lỗi hiện ghi nhận missing_features và model_not_loaded;
không đại diện cho mọi lỗi HTTP, validation hoặc exception có thể xảy ra.

## Kiểm tra cấu hình

```powershell
docker compose exec -T prometheus promtool check config /etc/prometheus/prometheus.yml
docker compose exec -T prometheus promtool check rules /etc/prometheus/alerts/model.yml
```

Trong Prometheus Targets, job wdbc-api phải UP.
promtool kiểm tra cấu hình; các thí nghiệm bên dưới kiểm chứng hành vi.

## Traffic và alert

Chạy từng lệnh riêng, quan sát dashboard và trang Prometheus Alerts.

```powershell
# Normal
docker compose exec -T api python scripts/traffic.py --rps 20 --seconds 300

# Khoảng 20% request thiếu feature
docker compose exec -T api python scripts/traffic.py --rps 20 --seconds 480 --broken 0.2

# Cộng 3 độ lệch chuẩn vào mỗi feature
docker compose exec -T api python scripts/traffic.py --rps 20 --seconds 1020 --drift 3.0
```

--rps là tốc độ đặt mục tiêu; tốc độ thực tế thấp hơn do script gửi
tuần tự và chờ thêm sau mỗi request.

| Alert | Điều kiện | Thời gian duy trì |
|---|---|---|
| ApiDown | up của wdbc-api bằng 0 | 1 phút |
| ModelNotLoaded | wdbc_model_loaded bằng 0 | 2 phút |
| HighErrorRate | Error share > 5% | 5 phút |
| SlowPredictions | p95 > 50 ms | 10 phút |
| MalignantShareShift | Lệch baseline 37% trên 15 điểm phần trăm | 15 phút |

Baseline 37% là giả định của tutorial.
Alert về malignant share là tín hiệu cần điều tra, không chứng minh
accuracy giảm. Không có Alertmanager hoặc kênh gửi thông báo trong stack này.

Để kiểm tra ApiDown:

```powershell
docker compose stop api
# Đợi ít nhất 75 giây rồi kiểm tra Prometheus Alerts.
docker compose start api
docker compose up -d --wait --wait-timeout 120
```

## Kiểm tra cardinality

Script dùng FastAPI TestClient trong container riêng, gửi cùng dữ liệu
với ID mới cho mỗi request. API phục vụ trên cổng 18000 không bị đổi chế độ.

```powershell
docker compose run --rm --no-deps -e T04_TRAP=0 api python -m scripts.check_cardinality
docker compose run --rm --no-deps -e T04_TRAP=1 api python -m scripts.check_cardinality
```

| Chế độ | Sau 200 prediction | Sau 400 prediction | Payload sau 400 |
|---|---:|---:|---:|
| T04_TRAP=0 | 23 series | 23 series | 4.110 bytes |
| T04_TRAP=1 | 419 series | 819 series | 74.533 bytes |

Số series chỉ tính metric có prefix wdbc_; payload là toàn bộ /metrics.
Kích thước payload có thể thay đổi nhẹ giữa các lần chạy.

sample_id tạo series mới cho mỗi ID khi bật trap, gồm _total và _created.
Label production nên có tập giá trị nhỏ, được kiểm soát; không dùng ID.
Traffic gốc tái sử dụng 569 ID, còn script kiểm tra tạo ID mới liên tục
để minh họa rõ tác động cardinality.

## Kết quả kiểm chứng

| Thí nghiệm | Kết quả thực tế |
|---|---|
| API smoke test | Prediction trả 200; thiếu feature trả 422 |
| Normal | 5.511 request thành công, 0 lỗi |
| Broken | 8.882 request; 7.094 thành công, 1.788 lỗi |
| Broken alert | Error share cửa sổ 5 phút: 20,25%; HighErrorRate firing |
| Drift | 18.731 request thành công, 0 lỗi |
| Drift metric | Malignant share 100%; error share 0; p95 khoảng 2,43 ms |
| Drift alert | MalignantShareShift firing; SlowPredictions inactive |
| Dừng API | up=0; ApiDown firing; API phục hồi sau khi start |
| Cardinality | Cả hai chế độ đều CARDINALITY CHECK PASSED |

Drift làm phân phối câu trả lời thay đổi trong khi lỗi và latency vẫn
bình thường. Alert web-service thông thường không phát hiện được tín hiệu này.

ModelNotLoaded và SlowPredictions đã được kiểm tra cấu hình;
chưa chủ động tạo tình huống để kiểm chứng firing.

## Thay đổi so với source nền

- Sửa annotation MalignantShareShift để mô tả đúng độ lệch baseline.
- Thêm scripts/check_cardinality.py với assertion cho hai chế độ.
- Bổ sung hướng dẫn Docker và kết quả thực nghiệm trong README.

## Dừng hệ thống

```powershell
docker compose down
```

Lệnh trên giữ dữ liệu Prometheus và Grafana trong Docker volumes.
