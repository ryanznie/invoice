import unittest
from unittest.mock import MagicMock
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import inference


class TestGeminiFallback(unittest.TestCase):
    def setUp(self):
        # Setup mocks
        self.mock_backend = MagicMock()
        self.mock_gemini = MagicMock()

        # Inject mocks into inference module
        inference.backend = self.mock_backend
        inference.gemini_client = self.mock_gemini
        inference.processor = MagicMock()  # Needs to be something

        # Mock inputs
        self.image = MagicMock()
        self.words = ["INVOICE", "#", "12345"]
        self.boxes = [[0, 0, 10, 10]] * 3

    def test_fallback_success(self):
        # Configure backend to fail
        self.mock_backend.predict.side_effect = Exception("Backend Crash!")

        # Configure Gemini to succeed
        self.mock_gemini.predict.return_value = {
            "invoice_number": "12345",
            "raw_response": "12345",
            "latency_ms": 100,
        }

        # Run prediction
        print("\nTesting Fallback Success...")
        result = inference.predict_invoice(self.image, self.words, self.boxes)

        # Verify result
        print(f"Result: {result}")
        self.assertEqual(result["invoice_number"], "12345")
        self.assertEqual(result["method"], "gemini")

        # Verify calls
        self.mock_backend.predict.assert_called_once()
        self.mock_gemini.predict.assert_called_once()
        print("✅ Fallback success verified")

    def test_fallback_failure(self):
        # Configure backend to fail
        self.mock_backend.predict.side_effect = Exception("Backend Crash!")

        # Configure Gemini to fail too
        self.mock_gemini.predict.return_value = {
            "invoice_number": None,
            "error": "Gemini Error",
        }

        print("\nTesting Fallback Failure...")
        with self.assertRaises(Exception) as cm:
            inference.predict_invoice(self.image, self.words, self.boxes)

        print(f"Caught expected exception: {cm.exception}")
        self.assertEqual(str(cm.exception), "Backend Crash!")
        print("✅ Fallback failure verified (properly re-raises original error)")


if __name__ == "__main__":
    unittest.main()
