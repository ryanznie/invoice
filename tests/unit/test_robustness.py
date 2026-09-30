"""
Tests for system robustness including network failures, timeouts, and shape mismatches
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.inference import TritonBackend, OnnxBackend

from PIL import Image


class TestTritonRobustness:
    """Tests for Triton Inference Server failure modes"""

    def test_triton_connection_refused(self):
        """Test handling when Triton server is unreachable"""
        backend = TritonBackend()

        # Mock connection error
        with patch("tritonclient.http.InferenceServerClient") as mock_client:
            mock_client.side_effect = ConnectionError("Connection refused")

            with pytest.raises(ValueError, match="Failed to create Triton client"):
                backend.predict({"input": np.array([1])})

    def test_triton_server_not_live(self):
        """Test handling when Triton server connects but is not live"""
        backend = TritonBackend()

        with patch("tritonclient.http.InferenceServerClient") as mock_client_cls:
            mock_client = Mock()
            mock_client.is_server_live.return_value = False
            mock_client_cls.return_value = mock_client

            with pytest.raises(ConnectionError, match="Triton server is not live"):
                backend.load("dummy/path")

    def test_triton_model_not_ready(self):
        """Test handling when Triton server is live but model is not ready"""
        backend = TritonBackend()

        with patch("tritonclient.http.InferenceServerClient") as mock_client_cls:
            mock_client = Mock()
            mock_client.is_server_live.return_value = True
            mock_client.is_server_ready.return_value = True
            mock_client.is_model_ready.return_value = False
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValueError, match="Model .* is not ready"):
                backend.load("dummy/path")

    def test_triton_inference_timeout(self):
        """Test handling of inference timeout"""
        backend = TritonBackend()

        with patch("tritonclient.http.InferenceServerClient") as mock_client_cls:
            mock_client = Mock()
            # Simulate generic exception that might occur on timeout (lib specific)
            mock_client.infer.side_effect = Exception("Deadline Exceeded")
            mock_client_cls.return_value = mock_client

            # Predict creates a new client
            with pytest.raises(Exception, match="Deadline Exceeded"):
                backend.predict({"input": np.zeros((1, 1))})


class TestOnnxRobustness:
    """Tests for ONNX Runtime failure modes"""

    def test_onnx_shape_mismatch(self):
        """Test handling of input shape mismatch"""
        backend = OnnxBackend()
        backend.session = Mock()

        # logical error from ORT when shapes don't match model definition
        backend.session.run.side_effect = Exception("Got invalid dimensions for input")

        with pytest.raises(Exception, match="Got invalid dimensions for input"):
            backend.predict({"input": np.zeros((1, 5))})  # Wrong shape

    def test_onnx_type_mismatch(self):
        """Test handling of input type mismatch"""
        backend = OnnxBackend()
        backend.session = Mock()

        backend.session.run.side_effect = Exception("Unexpected input data type")

        with pytest.raises(Exception, match="Unexpected input data type"):
            # Intentionally pass wrong type if we could, but here mocking exception
            backend.predict({"input": np.array(["string"])})

    def test_onnx_load_failure(self):
        """Test graceful failure when ONNX model is corrupt or incompatible"""
        backend = OnnxBackend()

        with patch("onnxruntime.InferenceSession") as mock_session:
            mock_session.side_effect = Exception(
                "Load model failed: protobuf parsing failed"
            )

            with pytest.raises(Exception, match="Load model failed"):
                backend.load("corrupt_model.onnx")


class TestAppRobustness:
    """Integration level robustness tests"""

    @patch("src.inference.backend")
    @patch("src.inference.processor")
    def test_predict_invoice_backend_failure(self, mock_processor, mock_backend):
        """Test that predict_invoice bubbles up backend errors correctly"""
        from src.inference import predict_invoice

        # Setup valid inputs
        mock_encoding = Mock()
        mock_encoding.word_ids.return_value = [0]
        mock_encoding.__getitem__ = Mock(return_value=np.array([0]))
        mock_processor.return_value = mock_encoding

        # Simulate backend failure
        mock_backend.predict.side_effect = RuntimeError("Backend crashed")

        with pytest.raises(RuntimeError, match="Backend crashed"):
            predict_invoice(Image.new("RGB", (100, 100)), ["test"], [[0, 0, 10, 10]])
