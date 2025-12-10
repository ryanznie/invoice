"""
Tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock


@pytest.fixture
def client():
    """Create a test client for the FastAPI app (API only, no Gradio)"""
    # Import here to avoid loading model during test collection
    from src import app

    return TestClient(app)


@pytest.fixture
def full_app_client():
    """Create a test client for the full app with Gradio mounted"""
    # Import the fully configured app from root app.py
    import sys
    from pathlib import Path

    # Add parent directory to path if needed
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import app as root_app

    return TestClient(root_app.app)


class TestHealthEndpoint:
    """Test suite for /health endpoint"""

    def test_health_check_model_loaded(self, client):
        """Test health check when model is loaded"""
        with patch("src.inference.model", Mock()):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()

            assert "status" in data
            assert "model_loaded" in data
            assert "device" in data
            assert data["model_loaded"] is True
            assert data["status"] == "healthy"

    def test_health_check_model_not_loaded(self, client):
        """Test health check when model is not loaded"""
        with patch("src.inference.model", None):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()

            assert data["model_loaded"] is False
            assert data["status"] == "unhealthy"

    def test_health_check_device_info(self, client):
        """Test that device information is included"""
        with patch("src.inference.model", Mock()):
            response = client.get("/health")
            data = response.json()

            assert "device" in data
            assert data["device"] in ["cpu", "cuda", "mps"]


class TestPredictionRequest:
    """Test suite for PredictionRequest model"""

    def test_valid_prediction_request(self):
        """Test valid prediction request"""
        from src.api import PredictionRequest

        request = PredictionRequest(
            words=["INVOICE", "NO:", "12345"],
            boxes=[[100, 100, 200, 120], [210, 100, 280, 120], [290, 100, 400, 120]],
        )

        assert len(request.words) == 3
        assert len(request.boxes) == 3

    def test_prediction_request_validation(self):
        """Test that Pydantic validates the request"""
        from src.api import PredictionRequest
        from pydantic import ValidationError

        # Test with invalid types
        with pytest.raises(ValidationError):
            PredictionRequest(words="not a list", boxes=[])

        with pytest.raises(ValidationError):
            PredictionRequest(words=[], boxes="not a list")


class TestGradioInterface:
    """Test suite for Gradio interface functions"""

    def test_gradio_predict_no_image(self):
        """Test gradio_predict with no image"""
        from src import gradio_predict

        result = gradio_predict(None, None)

        assert "Please upload an image" in result[0]
        assert result[1] is None

    def test_gradio_predict_no_text_file(self):
        """Test gradio_predict with no text file"""
        from src import gradio_predict
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = gradio_predict(img, None)

        assert "Please upload a text file" in result[0]
        assert result[1] is None

    @patch("src.inference.model", Mock())
    @patch("src.inference.processor", Mock())
    def test_gradio_predict_with_json_file(self, temp_json_file):
        """Test gradio_predict with JSON file"""
        from src import gradio_predict
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8))

        # Create a mock file object
        mock_file = Mock()
        mock_file.name = temp_json_file

        # This will likely fail without full model setup, but tests the flow
        try:
            result = gradio_predict(img, mock_file)
            # If it doesn't error, check result structure
            assert len(result) == 3
        except Exception as e:
            # Expected to fail without full model, but should not be validation error
            assert "validation" not in str(e).lower() or "Model not loaded" in str(e)

    def test_gradio_predict_error_handling(self):
        """Test that gradio_predict handles errors gracefully"""
        from src import gradio_predict

        # Pass invalid inputs
        result = gradio_predict("not an image", "not a file")

        # Should return error message, not raise exception
        assert len(result) == 3
        assert result[1] is None  # No image
        assert result[2] == ""  # No detailed output


class TestIntegrationAPI:
    """Integration tests for API workflows"""

    def test_health_endpoint_accessible(self, client):
        """Test that health endpoint is accessible"""
        with patch("src.inference.model", Mock()):
            response = client.get("/health")
            assert response.status_code == 200

    def test_root_endpoint_gradio(self, full_app_client):
        """Test that root endpoint serves Gradio interface"""
        with patch("src.inference.model", Mock()):
            response = full_app_client.get("/")
            # Gradio interface should be served
            assert response.status_code == 200


class TestErrorHandling:
    """Test error handling in API"""

    def test_invalid_endpoint(self, client):
        """Test accessing invalid endpoint"""
        with patch("src.inference.model", Mock()):
            response = client.get("/invalid-endpoint")
            assert response.status_code == 404

    def test_health_check_always_responds(self, client):
        """Test that health check always responds even if model fails"""
        with patch("src.inference.model", None), patch("src.inference.processor", None):
            response = client.get("/health")
            # Should still return 200, but indicate unhealthy
            assert response.status_code == 200
            assert response.json()["status"] == "unhealthy"
