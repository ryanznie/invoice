import os
import logging
import time
from typing import Dict, Optional, List, Any
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from benchmarks.models.base import BaseInvoiceModel, InferenceResult

load_dotenv()

logger = logging.getLogger(__name__)


class GeminiModel(BaseInvoiceModel):
    """
    Invoice extraction using Google's Gemini Flash model.
    """

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Gemini model.

        Args:
            model_config: Configuration dictionary. Can include:
                - model_name: Specific Gemini model version (default: gemini-2.5-flash)
                - api_key: Google Cloud API Key (optional, can use env var)
        """
        super().__init__(model_config)
        self.api_key = self.model_config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        # Default to gemini-2.5-flash as requested (assuming 2025 context)
        self.model_name = self.model_config.get("model_name", "gemini-2.5-flash-lite")
        self.model = None

    def load(self) -> None:
        """
        Load/Configure the Gemini API client.
        """
        if not genai:
            raise ImportError(
                "google-generativeai package is not installed. "
                "Please install it with: pip install google-generativeai"
            )

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set and not provided in config"
            )

        logger.info(f"Configuring Gemini with model: {self.model_name}")
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"✓ Gemini model initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise

    def predict(
        self,
        image: Optional[Image.Image] = None,
        words: Optional[List[str]] = None,
        boxes: Optional[List[List[int]]] = None,
        text: Optional[str] = None,
        **kwargs,
    ) -> InferenceResult:
        """
        Run inference using Gemini API.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

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

        # if text:
        #     inputs.append(f"\nDocument Text:\n{text}")
        elif words:
            # Reconstruct text from words if raw text not provided
            reconstructed_text = " ".join(words)
            inputs.append(f"\nDocument Text:\n{reconstructed_text}")

        if not image and not text and not words:
            raise ValueError("Gemini model requires image, text, or words input")

        start_time = time.time()
        # Retry logic parameters
        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries + 1):
            try:
                # Generation config for more deterministic output
                generation_config = genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=400,
                    temperature=0.0,  # Low temperature for factual extraction
                )

                response = self.model.generate_content(
                    inputs, generation_config=generation_config
                )

                # Handle potential safety blocks or empty responses
                if not response.text:
                    logger.warning(f"Empty response from Gemini for {self.model_name}")
                    return InferenceResult(
                        invoice_number=None,
                        confidence=0.0,
                        method=self.model_name,
                        metadata={
                            "error": "Empty response",
                            "safety_ratings": response.prompt_feedback,
                        },
                    )

                # If successful, break format loop
                break

            except Exception as e:
                if attempt < max_retries:
                    sleep_time = retry_delay * (2**attempt)
                    logger.warning(
                        f"Gemini inference failed (attempt {attempt+1}/{max_retries}). Retrying in {sleep_time}s... Error: {e}"
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"Gemini inference failed after {max_retries} retries: {e}"
                    )
                    return InferenceResult(
                        invoice_number=None,
                        confidence=0.0,
                        method=f"{self.model_name}_failed",
                        metadata={"error": str(e)},
                    )

        raw_text = response.text.strip()

        # Post-processing to clean up the result
        invoice_number = raw_text

        # diverse null checks
        if invoice_number.lower() in ["null", "none", "n/a", "not found"]:
            invoice_number = None
        else:
            # Remove common prefixes if the model ignored instructions
            prefixes = ["invoice number:", "invoice #:", "inv #:", "no.", "invoice no."]
            for prefix in prefixes:
                if invoice_number.lower().startswith(prefix):
                    invoice_number = invoice_number[len(prefix) :].strip()

        return InferenceResult(
            invoice_number=invoice_number,
            confidence=(
                1.0 if invoice_number else 0.0
            ),  # API doesn't give confidence per token easily
            method=self.model_name,
            metadata={
                "raw_response": raw_text,
                "model_name": self.model_name,
                "latency_ms": (time.time() - start_time) * 1000,
            },
        )

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            "model_name": self.model_name,
            "provider": "google_gemini",
        }
