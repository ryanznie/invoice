# Monitoring and Traffic Generation

This guide covers:

- generating load with real SROIE samples
- scraping Triton metrics with Prometheus
- building useful Grafana panels for inference serving

## Overview

In this project, the main monitoring target is Triton Inference Server, not the FastAPI app.

- Triton HTTP inference: `localhost:8000`
- Triton metrics: `localhost:8002/metrics`
- Prometheus UI: `http://localhost:9090`
- Grafana UI: `http://localhost:3000`

When Prometheus runs inside Docker Compose, it should scrape:

```yaml
- job_name: 'tritonserver'
  metrics_path: /metrics
  static_configs:
    - targets: ['tritonserver:8002']
```

Do not use `localhost:8002` inside the Prometheus container unless Prometheus itself is running on the host.

## Start the Monitoring Stack

Run the full stack:

```bash
docker compose up -d --build
```

Then verify:

```bash
curl http://localhost:7860/health
curl http://localhost:8002/metrics
```

Open Prometheus targets:

```text
http://localhost:9090/targets
```

The `tritonserver` target should be `UP`.

## Generate Traffic with Real Data

To make Prometheus and Grafana useful, you need sustained inference traffic. Use the real SROIE dataset already present in:

- `data/SROIE2019/test/img`
- `data/SROIE2019/test/box`

### Quick Single Request

```bash
curl -X POST http://localhost:7860/predict \
  -F "image=@data/SROIE2019/test/img/X51005675104.jpg" \
  -F "ocr_file=@data/SROIE2019/test/box/X51005675104.txt"
```

### Simple Replay Loop

This sends repeated requests using real image and OCR pairs:

```bash
for f in data/SROIE2019/test/img/*.jpg; do
  base=$(basename "$f" .jpg)
  curl -s -X POST http://localhost:7860/predict \
    -F "image=@${f}" \
    -F "ocr_file=@data/SROIE2019/test/box/${base}.txt" > /dev/null
done
```

### Sustained Load

Use a loop to keep traffic flowing while watching dashboards:

```bash
while true; do
  for f in data/SROIE2019/test/img/*.jpg; do
    base=$(basename "$f" .jpg)
    curl -s -X POST http://localhost:7860/predict \
      -F "image=@${f}" \
      -F "ocr_file=@data/SROIE2019/test/box/${base}.txt" > /dev/null
  done
done
```

If you want higher throughput, run multiple terminals with the same loop.

## Prometheus: Useful Triton Metrics

The main Triton metrics exposed by this stack are:

- `nv_inference_request_success`
- `nv_inference_request_failure`
- `nv_inference_count`
- `nv_inference_exec_count`
- `nv_inference_request_duration_us`
- `nv_inference_queue_duration_us`
- `nv_inference_compute_input_duration_us`
- `nv_inference_compute_infer_duration_us`
- `nv_inference_compute_output_duration_us`
- `nv_inference_pending_request_count`
- `nv_cpu_utilization`
- `nv_cpu_memory_total_bytes`
- `nv_cpu_memory_used_bytes`

These duration metrics are cumulative counters, so use `rate(...)` and divide by request volume to get average durations.

## Grafana Panels

Create a dashboard in Grafana and add the following panels with the `Prometheus` datasource.

### Request Rate

Type: Time series

```promql
sum(rate(nv_inference_request_success[1m])) by (model, version)
```

Unit: `req/s`

### Failure Rate

Type: Time series

```promql
sum(rate(nv_inference_request_failure[1m])) by (model, version)
```

Unit: `req/s`

### Error Percentage

Type: Time series

```promql
100 * sum(rate(nv_inference_request_failure[5m])) by (model, version)
/
clamp_min(
  sum(rate(nv_inference_request_success[5m]) + rate(nv_inference_request_failure[5m])) by (model, version),
  1e-9
)
```

Unit: `percent (0-100)`

### Average Request Latency

Type: Time series

```promql
(
  sum(rate(nv_inference_request_duration_us[5m])) by (model, version)
  /
  clamp_min(sum(rate(nv_inference_request_success[5m])) by (model, version), 1e-9)
) / 1000
```

Unit: `ms`

### Average Queue Latency

Type: Time series

```promql
(
  sum(rate(nv_inference_queue_duration_us[5m])) by (model, version)
  /
  clamp_min(sum(rate(nv_inference_request_success[5m])) by (model, version), 1e-9)
) / 1000
```

Unit: `ms`

### Average Input Time

Type: Time series

```promql
(
  sum(rate(nv_inference_compute_input_duration_us[5m])) by (model, version)
  /
  clamp_min(sum(rate(nv_inference_count[5m])) by (model, version), 1e-9)
) / 1000
```

Unit: `ms`

### Average Infer Time

Type: Time series

```promql
(
  sum(rate(nv_inference_compute_infer_duration_us[5m])) by (model, version)
  /
  clamp_min(sum(rate(nv_inference_count[5m])) by (model, version), 1e-9)
) / 1000
```

Unit: `ms`

### Average Output Time

Type: Time series

```promql
(
  sum(rate(nv_inference_compute_output_duration_us[5m])) by (model, version)
  /
  clamp_min(sum(rate(nv_inference_count[5m])) by (model, version), 1e-9)
) / 1000
```

Unit: `ms`

### Pending Requests

Type: Time series

```promql
nv_inference_pending_request_count
```

### Inferences Per Second

Type: Time series

```promql
sum(rate(nv_inference_count[1m])) by (model, version)
```

Unit: `infer/s`

### Execution Rate

Type: Time series

```promql
sum(rate(nv_inference_exec_count[1m])) by (model, version)
```

Unit: `exec/s`

### CPU Utilization

Type: Time series

```promql
nv_cpu_utilization * 100
```

Unit: `percent (0-100)`

### CPU Memory Used

Type: Time series

```promql
nv_cpu_memory_used_bytes
```

Unit: `bytes`

### CPU Memory Utilization

Type: Time series

```promql
100 * nv_cpu_memory_used_bytes / nv_cpu_memory_total_bytes
```

Unit: `percent (0-100)`

## Suggested Dashboard Layout

Top row:

- request rate
- failure rate
- error percentage
- pending requests

Middle row:

- average request latency
- average queue latency
- average infer time

Bottom row:

- CPU utilization
- CPU memory used
- CPU memory utilization

## Notes

- These metrics support averages, not p95/p99 latency, because Triton is exporting cumulative duration counters rather than histogram buckets.
- If request rate rises and queue latency rises first, the server is saturating before model compute becomes the bottleneck.
- Save important Grafana dashboards back into `monitoring/grafana/dashboards/` if you want them version-controlled.
