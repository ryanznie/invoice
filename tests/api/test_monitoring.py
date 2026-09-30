"""
Tests for Prometheus monitoring instrumentation.
"""

from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client():
    from src import app

    with patch("src.api.inference.load_model", Mock()):
        yield TestClient(app)


def _metric_value(
    metrics: str, name: str, labels: dict[str, str] | None = None
) -> float:
    labels = labels or {}
    for line in metrics.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        metric_name, raw_value = line.rsplit(" ", 1)
        if labels:
            expected = ",".join(f'{key}="{value}"' for key, value in labels.items())
            if "{" not in metric_name or expected not in metric_name:
                continue
        elif "{" in metric_name:
            continue
        return float(raw_value)
    return 0.0


def _image_bytes() -> bytes:
    image = Image.new("RGB", (800, 600), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestPrometheusMetrics:
    def test_metrics_endpoint_exposes_expected_series(self, client):
        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        body = response.text
        assert "inference_requests_total" in body
        assert "inference_latency_seconds" in body
        assert "model_inference_latency_seconds" in body
        assert "inference_errors_total" in body

    def test_runtime_config_exposes_model_serving_details(self, client):
        response = client.get("/runtime/config")

        assert response.status_code == 200
        body = response.json()
        assert body["inference_backend"]
        assert body["base_model"]
        assert "model_loaded" in body

    @patch("src.inference.backend", Mock())
    @patch("src.inference.processor", Mock())
    def test_predict_success_increments_metrics(self, client, temp_json_file):
        before = client.get("/metrics").text
        before_success = _metric_value(
            before,
            "inference_requests_total",
            {"method": "heuristic", "status": "success"},
        )

        response = client.post(
            "/predict",
            files={
                "image": ("invoice.png", _image_bytes(), "image/png"),
                "ocr_file": (
                    "ocr.json",
                    Path(temp_json_file).read_bytes(),
                    "application/json",
                ),
            },
        )

        after = client.get("/metrics").text
        after_success = _metric_value(
            after,
            "inference_requests_total",
            {"method": "heuristic", "status": "success"},
        )

        assert response.status_code == 200
        assert after_success == before_success + 1
        assert _metric_value(after, "inference_latency_seconds_count") >= 1

    def test_predict_error_increments_metrics(self, client, temp_json_file):
        before = client.get("/metrics").text
        before_error = _metric_value(
            before,
            "inference_requests_total",
            {"method": "unknown", "status": "error"},
        )

        with (
            patch("src.inference.backend", None),
            patch("src.inference.processor", None),
        ):
            response = client.post(
                "/predict",
                files={
                    "image": ("invoice.png", _image_bytes(), "image/png"),
                    "ocr_file": (
                        "ocr.json",
                        Path(temp_json_file).read_bytes(),
                        "application/json",
                    ),
                },
            )

        after = client.get("/metrics").text
        after_error = _metric_value(
            after,
            "inference_requests_total",
            {"method": "unknown", "status": "error"},
        )

        assert response.status_code == 503
        assert after_error == before_error + 1
