import os
import logging
import time
from typing import Optional, List, Dict, Any
from PIL import Image

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Client for interacting with Google's Gemini Flash model for invoice extraction.
    Used as a fallback when the primary model fails.
    """

    def __init__(
        self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Google Cloud API Key (optional, defaults to env var)
            model_name: Specific Gemini model version
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.client = None
        self._initialized = False

    def load(self) -> None:
        """
        Lazy load the Gemini API client.
        """
        if self._initialized:
            return

        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not set. Gemini fallback will not work.")
            return

        try:
            self.client = genai.Client(api_key=self.api_key)
            self._initialized = True
            logger.info(f"✓ Gemini client initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    def predict(
        self,
        image: Optional[Image.Image] = None,
        words: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference using Gemini API.

        Returns:
            Dict containing 'invoice_number', 'raw_response', and 'error'
        """
        if not self._initialized:
            self.load()

        if not self.client:
            return {"error": "Gemini client not initialized"}

        # Construct prompt
        system_prompt = (
            "You are an expert invoice extraction system. "
            "Your task is to extract the INVOICE NUMBER or a unique identifier on a receipt indicating transaction from the provided document image or text. "
            "Return ONLY the invoice number string. "
            "Do not include labels like 'Invoice #', 'Number:', etc. "
            "If the invoice number is not visible or cannot be found, return the string 'NULL'."
        )

        inputs = [system_prompt]

        # Add image or text context
        if image:
            inputs.append(image)
        elif words:
            # Reconstruct text from words if raw text not provided
            reconstructed_text = " ".join(words)
            inputs.append(f"\nDocument Text:\n{reconstructed_text}")
        else:
            return {"error": "No input provided (image or words)"}

        start_time = time.time()

        try:
            # Generation config for deterministic output
            config = types.GenerateContentConfig(
                candidate_count=1,
                max_output_tokens=400,
                temperature=0.0,
            )

            logger.info(f"Sending request to Gemini ({self.model_name})...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=inputs,
                config=config,
            )

            if not response.text:
                return {"invoice_number": None, "error": "Empty response"}

            raw_text = response.text.strip()
            invoice_number = self._clean_output(raw_text)

            latency = (time.time() - start_time) * 1000

            return {
                "invoice_number": invoice_number,
                "raw_response": raw_text,
                "latency_ms": latency,
                "method": "gemini_fallback",
            }

        except Exception as e:
            logger.error(f"Gemini inference failed: {e}")
            return {"invoice_number": None, "error": str(e)}

    def _clean_output(self, text: str) -> Optional[str]:
        """Clean up the model output."""
        if not text:
            return None

        cleaned = text.strip()

        # Check for null indicators
        if cleaned.lower() in ["null", "none", "n/a", "not found"]:
            return None

        # Remove common prefixes
        prefixes = ["invoice number:", "invoice #:", "inv #:", "no.", "invoice no."]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()

        return cleaned
