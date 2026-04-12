"""
Tests for FastAPI endpoints.
"""

from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from src import app

    return TestClient(app)


class TestHealthEndpoint:
    """Test suite for /health endpoint."""

    def test_health_check_model_loaded(self, client):
        with patch("src.inference.backend", Mock()):
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"
        assert data["device"] in ["cpu", "cuda", "mps"]

    def test_health_check_triton_backend(self, client):
        with patch("src.inference.model", None), patch("src.inference.backend", Mock()):
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"

    def test_health_check_model_not_loaded(self, client):
        with patch("src.inference.backend", None):
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is False
        assert data["status"] == "unhealthy"


class TestPredictionRequest:
    """Test suite for PredictionRequest model."""

    def test_valid_prediction_request(self):
        from pydantic import ValidationError
        from src.api import PredictionRequest

        request = PredictionRequest(
            words=["INVOICE", "NO:", "12345"],
            boxes=[[100, 100, 200, 120], [210, 100, 280, 120], [290, 100, 400, 120]],
        )

        assert len(request.words) == 3
        assert len(request.boxes) == 3

        with pytest.raises(ValidationError):
            PredictionRequest(words="not a list", boxes=[])


class TestIntegrationAPI:
    """Integration tests for API workflows."""

    def test_root_endpoint_metadata(self, client):
        with patch("src.inference.backend", Mock()):
            response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Invoice NER API"
        assert data["predict_url"] == "/predict"
        assert data["health_url"] == "/health"

    @patch("src.inference.backend", Mock())
    @patch("src.inference.processor", Mock())
    @patch("src.api.inference.predict_invoice")
    def test_predict_returns_boxes_for_frontend(
        self, mock_predict_invoice, client, temp_json_file
    ):
        mock_predict_invoice.return_value = {
            "invoice_number": "INV-12345",
            "labels": ["LABEL_0", "LABEL_0", "LABEL_1"],
            "confidence_scores": [0.98, 0.97, 0.96],
        }

        image = Image.new("RGB", (800, 600), color="white")
        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)

        with open(temp_json_file, "rb") as ocr_file:
            response = client.post(
                "/predict",
                files={
                    "image": ("invoice.png", image_buffer.read(), "image/png"),
                    "ocr_file": ("ocr.json", ocr_file.read(), "application/json"),
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["invoice_number"] == "INV-12345"
        assert data["image_size"] == {"width": 800, "height": 600}
        assert data["predictions"][2]["index"] == 2
        assert data["predictions"][2]["box"] == [290, 100, 400, 120]
        assert data["predictions"][2]["is_invoice_number"] is True


class TestErrorHandling:
    """Test error handling in API."""

    def test_invalid_endpoint(self, client):
        with patch("src.inference.backend", Mock()):
            response = client.get("/invalid-endpoint")

        assert response.status_code == 404

    def test_health_check_always_responds(self, client):
        with (
            patch("src.inference.backend", None),
            patch("src.inference.processor", None),
        ):
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "unhealthy"
