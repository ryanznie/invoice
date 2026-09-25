"""
Generate provisioned Grafana dashboards for Invoice NER.

Run:
    uv run python scripts/generate_grafana_dashboards.py
"""

from __future__ import annotations

import json
from pathlib import Path

DASHBOARD_DIR = Path("monitoring/grafana/dashboards")
DATASOURCE = "Prometheus"


def target(expr: str, legend: str = "") -> dict:
    result = {"expr": expr}
    if legend:
        result["legendFormat"] = legend
    return result


def grid(x: int, y: int, w: int, h: int) -> dict:
    return {"x": x, "y": y, "w": w, "h": h}


def stat(panel_id: int, title: str, pos: dict, expr: str, unit: str = "none") -> dict:
    return {
        "id": panel_id,
        "title": title,
        "type": "stat",
        "datasource": DATASOURCE,
        "gridPos": pos,
        "targets": [target(expr)],
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
    }


def timeseries(
    panel_id: int,
    title: str,
    pos: dict,
    targets: list[dict],
    unit: str = "none",
) -> dict:
    return {
        "id": panel_id,
        "title": title,
        "type": "timeseries",
        "datasource": DATASOURCE,
        "gridPos": pos,
        "targets": targets,
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
    }


def bar_gauge(panel_id: int, title: str, pos: dict, expr: str) -> dict:
    return {
        "id": panel_id,
        "title": title,
        "type": "bargauge",
        "datasource": DATASOURCE,
        "gridPos": pos,
        "targets": [target(expr, "{{method}} {{status}}")],
        "options": {"orientation": "horizontal", "displayMode": "gradient"},
    }


def heatmap(panel_id: int, title: str, pos: dict, expr: str) -> dict:
    return {
        "id": panel_id,
        "title": title,
        "type": "heatmap",
        "datasource": DATASOURCE,
        "gridPos": pos,
        "targets": [{"expr": expr, "format": "heatmap"}],
    }


def row(panel_id: int, title: str, y: int) -> dict:
    return {"id": panel_id, "title": title, "type": "row", "gridPos": grid(0, y, 24, 1)}


def dashboard(uid: str, title: str, panels: list[dict]) -> dict:
    return {
        "annotations": {"list": []},
        "editable": True,
        "graphTooltip": 0,
        "id": None,
        "links": [],
        "panels": panels,
        "refresh": "5s",
        "schemaVersion": 38,
        "style": "dark",
        "tags": ["invoice-ner", "production"],
        "templating": {"list": []},
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {},
        "timezone": "",
        "title": title,
        "uid": uid,
        "version": 1,
    }


def executive_overview() -> dict:
    return dashboard(
        "invoice-ner-executive",
        "Invoice NER - Executive Overview",
        [
            row(1, "SLO Snapshot", 0),
            stat(
                2,
                "Availability",
                grid(0, 1, 4, 4),
                '100 * sum(rate(inference_requests_total{status="success"}[5m])) / sum(rate(inference_requests_total[5m]))',
                "percent",
            ),
            stat(
                3,
                "Error Rate",
                grid(4, 1, 4, 4),
                '100 * sum(rate(inference_requests_total{status="error"}[5m])) / sum(rate(inference_requests_total[5m]))',
                "percent",
            ),
            stat(
                4,
                "P95 E2E Latency",
                grid(8, 1, 4, 4),
                "histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[5m])) by (le))",
                "s",
            ),
            stat(
                5,
                "Triton Compute Avg",
                grid(12, 1, 4, 4),
                "sum(rate(nv_inference_compute_infer_duration_us[5m])) / sum(rate(nv_inference_exec_count[5m])) / 1000000",
                "s",
            ),
            stat(
                6,
                "API Target Up",
                grid(16, 1, 4, 4),
                'up{job="invoice-ner-app"}',
            ),
            stat(
                7,
                "Triton Target Up",
                grid(20, 1, 4, 4),
                'up{job="tritonserver"}',
            ),
            timeseries(
                8,
                "Request Rate",
                grid(0, 5, 8, 7),
                [
                    target(
                        "sum(rate(inference_requests_total[1m])) by (method, status)",
                        "{{method}} {{status}}",
                    )
                ],
                "reqps",
            ),
            timeseries(
                9,
                "E2E Latency",
                grid(8, 5, 8, 7),
                [
                    target(
                        "histogram_quantile(0.50, sum(rate(inference_latency_seconds_bucket[5m])) by (le))",
                        "p50",
                    ),
                    target(
                        "histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[5m])) by (le))",
                        "p95",
                    ),
                    target(
                        "histogram_quantile(0.99, sum(rate(inference_latency_seconds_bucket[5m])) by (le))",
                        "p99",
                    ),
                ],
                "s",
            ),
            bar_gauge(
                10,
                "Method / Status Mix",
                grid(16, 5, 8, 7),
                "sum(increase(inference_requests_total[1h])) by (method, status)",
            ),
            timeseries(
                11,
                "Fallbacks / sec",
                grid(0, 12, 8, 7),
                [target("sum(rate(fallback_total[5m]))", "fallbacks/sec")],
                "ops",
            ),
            timeseries(
                13,
                "Triton Inference Activity",
                grid(8, 12, 8, 7),
                [
                    target("sum(rate(nv_inference_request_success[5m]))", "success"),
                    target("sum(rate(nv_inference_request_failure[5m]))", "failure"),
                ],
                "ops",
            ),
            timeseries(
                14,
                "API Model Latency Percentiles",
                grid(16, 12, 8, 7),
                [
                    target(
                        "histogram_quantile(0.50, sum(rate(model_inference_latency_seconds_bucket[5m])) by (le))",
                        "p50",
                    ),
                    target(
                        "histogram_quantile(0.95, sum(rate(model_inference_latency_seconds_bucket[5m])) by (le))",
                        "p95",
                    ),
                    target(
                        "histogram_quantile(0.99, sum(rate(model_inference_latency_seconds_bucket[5m])) by (le))",
                        "p99",
                    ),
                    target(
                        "sum(rate(nv_inference_compute_infer_duration_us[5m])) / sum(rate(nv_inference_exec_count[5m])) / 1000000",
                        "triton compute avg",
                    ),
                ],
                "s",
            ),
        ],
    )


def api_infra() -> dict:
    return dashboard(
        "invoice-ner-api-infra",
        "Invoice NER - API and Infra",
        [
            row(1, "API", 0),
            timeseries(
                2,
                "Request Rate by Method / Status",
                grid(0, 1, 12, 7),
                [
                    target(
                        "sum(rate(inference_requests_total[1m])) by (method, status)",
                        "{{method}} {{status}}",
                    )
                ],
                "reqps",
            ),
            timeseries(
                3,
                "Error Rate",
                grid(12, 1, 12, 7),
                [
                    target(
                        '100 * sum(rate(inference_requests_total{status="error"}[5m])) / sum(rate(inference_requests_total[5m]))',
                        "error %",
                    )
                ],
                "percent",
            ),
            timeseries(
                4,
                "E2E Latency Quantiles",
                grid(0, 8, 12, 7),
                [
                    target(
                        "histogram_quantile(0.50, sum(rate(inference_latency_seconds_bucket[5m])) by (le))",
                        "p50",
                    ),
                    target(
                        "histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[5m])) by (le))",
                        "p95",
                    ),
                    target(
                        "histogram_quantile(0.99, sum(rate(inference_latency_seconds_bucket[5m])) by (le))",
                        "p99",
                    ),
                ],
                "s",
            ),
            heatmap(
                5,
                "E2E Latency Distribution",
                grid(12, 8, 12, 7),
                "sum(rate(inference_latency_seconds_bucket[5m])) by (le)",
            ),
            timeseries(
                14,
                "API Model Latency Percentiles",
                grid(0, 15, 12, 7),
                [
                    target(
                        "histogram_quantile(0.50, sum(rate(model_inference_latency_seconds_bucket[5m])) by (le))",
                        "p50",
                    ),
                    target(
                        "histogram_quantile(0.95, sum(rate(model_inference_latency_seconds_bucket[5m])) by (le))",
                        "p95",
                    ),
                    target(
                        "histogram_quantile(0.99, sum(rate(model_inference_latency_seconds_bucket[5m])) by (le))",
                        "p99",
                    ),
                    target(
                        "sum(rate(nv_inference_compute_infer_duration_us[5m])) / sum(rate(nv_inference_exec_count[5m])) / 1000000",
                        "triton compute avg",
                    ),
                ],
                "s",
            ),
            heatmap(
                15,
                "API Model Latency Distribution",
                grid(12, 15, 12, 7),
                "sum(rate(model_inference_latency_seconds_bucket[5m])) by (le)",
            ),
            row(6, "Targets", 22),
            stat(7, "API Target Up", grid(0, 23, 6, 4), 'up{job="invoice-ner-app"}'),
            stat(8, "Triton Target Up", grid(6, 23, 6, 4), 'up{job="tritonserver"}'),
            timeseries(
                9,
                "Prometheus Scrape Duration",
                grid(12, 23, 12, 4),
                [target("scrape_duration_seconds", "{{job}}")],
                "s",
            ),
            row(10, "Triton", 27),
            timeseries(
                11,
                "Triton Request Rate",
                grid(0, 28, 8, 7),
                [
                    target("sum(rate(nv_inference_request_success[5m]))", "success"),
                    target("sum(rate(nv_inference_request_failure[5m]))", "failure"),
                ],
                "ops",
            ),
            timeseries(
                12,
                "Triton Queue / Compute Duration",
                grid(8, 28, 8, 7),
                [
                    target(
                        "sum(rate(nv_inference_queue_duration_us[5m])) / sum(rate(nv_inference_request_success[5m])) / 1000000",
                        "queue avg",
                    ),
                    target(
                        "sum(rate(nv_inference_compute_infer_duration_us[5m])) / sum(rate(nv_inference_request_success[5m])) / 1000000",
                        "compute avg",
                    ),
                ],
                "s",
            ),
            timeseries(
                13,
                "Triton GPU Utilization",
                grid(16, 28, 8, 7),
                [target("nv_gpu_utilization", "gpu util")],
                "percent",
            ),
        ],
    )


def model_behavior() -> dict:
    return dashboard(
        "invoice-ner-model",
        "Invoice NER - Model Behavior",
        [
            row(1, "Extraction Behavior", 0),
            bar_gauge(
                2,
                "Extraction Method / Status Counts",
                grid(0, 1, 8, 7),
                "sum(increase(inference_requests_total[1h])) by (method, status)",
            ),
            timeseries(
                3,
                "Fallback Rate",
                grid(8, 1, 8, 7),
                [target("sum(rate(fallback_total[5m]))", "fallbacks/sec")],
                "ops",
            ),
            timeseries(
                4,
                "Model Request Rate",
                grid(16, 1, 8, 7),
                [
                    target(
                        'sum(rate(inference_requests_total{method="model",status="success"}[5m]))',
                        "model success",
                    )
                ],
                "ops",
            ),
            timeseries(
                6,
                "API Model Latency Percentiles",
                grid(0, 8, 12, 8),
                [
                    target(
                        "histogram_quantile(0.50, sum(rate(model_inference_latency_seconds_bucket[5m])) by (le))",
                        "p50",
                    ),
                    target(
                        "histogram_quantile(0.95, sum(rate(model_inference_latency_seconds_bucket[5m])) by (le))",
                        "p95",
                    ),
                    target(
                        "histogram_quantile(0.99, sum(rate(model_inference_latency_seconds_bucket[5m])) by (le))",
                        "p99",
                    ),
                    target(
                        "sum(rate(nv_inference_compute_infer_duration_us[5m])) / sum(rate(nv_inference_exec_count[5m])) / 1000000",
                        "triton compute avg",
                    ),
                ],
                "s",
            ),
            timeseries(
                11,
                "Prediction Volume by Method",
                grid(12, 8, 12, 8),
                [
                    target(
                        'sum(rate(inference_requests_total{status="success"}[5m])) by (method)',
                        "{{method}}",
                    )
                ],
                "ops",
            ),
            row(7, "Metrics To Add", 16),
            stat(
                8,
                "Not Found Rate (Add Metric)",
                grid(0, 17, 8, 4),
                "0",
                "percent",
            ),
            stat(
                9,
                "Valid Format Rate (Add Metric)",
                grid(8, 17, 8, 4),
                "0",
                "percent",
            ),
            stat(
                10,
                "OCR Parse Error Rate (Add Metric)",
                grid(16, 17, 8, 4),
                "0",
                "percent",
            ),
        ],
    )


def load_test() -> dict:
    return dashboard(
        "invoice-ner-load-test",
        "Invoice NER - Load Test",
        [
            row(1, "During Locust Runs", 0),
            stat(
                2,
                "Current RPS",
                grid(0, 1, 4, 4),
                "sum(rate(inference_requests_total[1m]))",
                "reqps",
            ),
            stat(
                3,
                "Predict RPS",
                grid(4, 1, 4, 4),
                "sum(rate(inference_requests_total[1m]))",
                "reqps",
            ),
            stat(
                4,
                "Failure Rate",
                grid(8, 1, 4, 4),
                '100 * sum(rate(inference_requests_total{status="error"}[5m])) / sum(rate(inference_requests_total[5m]))',
                "percent",
            ),
            stat(
                5,
                "P99 E2E Latency",
                grid(12, 1, 4, 4),
                "histogram_quantile(0.99, sum(rate(inference_latency_seconds_bucket[5m])) by (le))",
                "s",
            ),
            stat(
                6,
                "Triton Target",
                grid(16, 1, 4, 4),
                'up{job="tritonserver"}',
            ),
            stat(
                7,
                "API Target",
                grid(20, 1, 4, 4),
                'up{job="invoice-ner-app"}',
            ),
            timeseries(
                8,
                "Throughput",
                grid(0, 5, 8, 7),
                [
                    target(
                        "sum(rate(inference_requests_total[1m])) by (method, status)",
                        "{{method}} {{status}}",
                    )
                ],
                "reqps",
            ),
            timeseries(
                9,
                "E2E Latency Under Load",
                grid(8, 5, 8, 7),
                [
                    target(
                        "histogram_quantile(0.50, sum(rate(inference_latency_seconds_bucket[1m])) by (le))",
                        "p50",
                    ),
                    target(
                        "histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[1m])) by (le))",
                        "p95",
                    ),
                    target(
                        "histogram_quantile(0.99, sum(rate(inference_latency_seconds_bucket[1m])) by (le))",
                        "p99",
                    ),
                ],
                "s",
            ),
            timeseries(
                13,
                "API Model Latency Under Load",
                grid(16, 12, 8, 8),
                [
                    target(
                        "histogram_quantile(0.50, sum(rate(model_inference_latency_seconds_bucket[1m])) by (le))",
                        "p50",
                    ),
                    target(
                        "histogram_quantile(0.95, sum(rate(model_inference_latency_seconds_bucket[1m])) by (le))",
                        "p95",
                    ),
                    target(
                        "histogram_quantile(0.99, sum(rate(model_inference_latency_seconds_bucket[1m])) by (le))",
                        "p99",
                    ),
                    target(
                        "sum(rate(nv_inference_compute_infer_duration_us[1m])) / sum(rate(nv_inference_exec_count[1m])) / 1000000",
                        "triton compute avg",
                    ),
                ],
                "s",
            ),
            timeseries(
                10,
                "Triton Request Rate",
                grid(16, 5, 8, 7),
                [
                    target(
                        "sum(rate(nv_inference_request_success[1m]))", "triton success"
                    )
                ],
                "ops",
            ),
            heatmap(
                11,
                "Latency Heatmap",
                grid(0, 12, 12, 8),
                "sum(rate(inference_latency_seconds_bucket[1m])) by (le)",
            ),
        ],
    )


def main() -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    dashboards = {
        "executive_overview.json": executive_overview(),
        "api_infra.json": api_infra(),
        "model_behavior.json": model_behavior(),
        "load_test.json": load_test(),
    }
    for filename, payload in dashboards.items():
        (DASHBOARD_DIR / filename).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
