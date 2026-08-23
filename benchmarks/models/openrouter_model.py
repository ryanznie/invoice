"""
OpenRouter model implementation for invoice extraction benchmarks.
"""

import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from PIL import Image

from benchmarks.models.base import BaseInvoiceModel, InferenceResult

load_dotenv()

logger = logging.getLogger(__name__)


class OpenRouterModel(BaseInvoiceModel):
    """Invoice extraction using a hosted OpenRouter vision model."""

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_config)
        self.client = None
        self.api_key = self.model_config.get("api_key") or os.getenv(
            "OPENROUTER_API_KEY"
        )
        self.model_name = (
            self.model_config.get("model_path")
            or os.getenv("OPENROUTER_MODEL")
            or "qwen/qwen2.5-vl-72b-instruct"
        )
        self.base_url = self.model_config.get(
            "base_url",
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        self.max_tokens = int(
            self.model_config.get(
                "max_new_tokens", os.getenv("OPENROUTER_MAX_TOKENS", "128")
            )
        )

    def load(self) -> None:
        """Load/configure the OpenRouter API client."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info("OpenRouter client initialized for model: %s", self.model_name)

    def predict(
        self,
        image: Optional[Image.Image] = None,
        words: Optional[List[str]] = None,
        boxes: Optional[List[List[int]]] = None,
        text: Optional[str] = None,
        **kwargs,
    ) -> InferenceResult:
        """Run inference using OpenRouter."""
        if self.client is None:
            raise RuntimeError("Client not loaded. Call load() first.")
        if image is None and not text and not words:
            raise ValueError("OpenRouter model requires image, text, or words input")

        prompt = (
            "Extract the invoice number from this invoice. "
            "Return only valid JSON with this exact schema: "
            '{"invoice_number": string | null}. '
            "Use null when no invoice number is visible. "
            "Do not return dates, totals, tax IDs, phone numbers, or receipt line items."
        )
        if text:
            prompt = f"{prompt}\n\nOCR text:\n{text}"
        elif words:
            prompt = f"{prompt}\n\nOCR text:\n{' '.join(words)}"

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(image)},
                }
            )

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content or ""
            invoice_number = self._clean_output(raw_text)
            metadata = {
                "raw_response": raw_text,
                "model_name": self.model_name,
                "latency_ms": (time.time() - start_time) * 1000,
            }
            if response.usage:
                metadata["usage"] = response.usage.model_dump()

            return InferenceResult(
                invoice_number=invoice_number,
                confidence=1.0 if invoice_number else 0.0,
                method=self.model_name,
                metadata=metadata,
            )
        except Exception as exc:
            logger.error("OpenRouter inference failed: %s", exc)
            return InferenceResult(
                invoice_number=None,
                confidence=0.0,
                method=self.model_name,
                metadata={
                    "error": str(exc),
                    "model_name": self.model_name,
                    "failed": True,
                },
            )

    def _image_to_data_url(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _clean_output(self, text: str) -> Optional[str]:
        if not text:
            return None

        cleaned = text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fenced_match:
            cleaned = fenced_match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, dict):
            return None

        value = parsed.get("invoice_number")
        if value is None or not isinstance(value, str):
            return None

        cleaned = value.strip()

        if cleaned.lower() in {"", "null", "none", "n/a", "not found"}:
            return None
        return cleaned

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            "model_name": self.model_name,
            "provider": "openrouter",
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
        }
