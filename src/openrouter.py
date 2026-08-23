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

load_dotenv()

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """
    Client for hosted VLM invoice extraction through OpenRouter.

    Used as a fallback when the primary local model fails.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = (
            model_name
            or os.getenv("OPENROUTER_MODEL")
            or "qwen/qwen2.5-vl-72b-instruct"
        )
        self.base_url = base_url or os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "128"))
        self.client = None
        self._initialized = False

    def load(self) -> None:
        """Lazy load the OpenRouter API client."""
        if self._initialized:
            return

        if not self.api_key:
            logger.warning(
                "OPENROUTER_API_KEY not set. OpenRouter fallback will not work."
            )
            return

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._initialized = True
            logger.info("OpenRouter client initialized with model: %s", self.model_name)
        except Exception as exc:
            logger.error("Failed to initialize OpenRouter client: %s", exc)

    def predict(
        self,
        image: Optional[Image.Image] = None,
        words: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run invoice extraction using the configured OpenRouter VLM.

        Returns:
            Dict containing 'invoice_number', 'raw_response', and 'error'
        """
        if not self._initialized:
            self.load()

        if not self.client:
            return {"error": "OpenRouter client not initialized"}

        if image is None and not words:
            return {"error": "No input provided (image or words)"}

        prompt = (
            "Extract the invoice number from this invoice. "
            "Return only valid JSON with this exact schema: "
            '{"invoice_number": string | null}. '
            "Use null when no invoice number is visible. "
            "Do not return dates, totals, tax IDs, phone numbers, or receipt line items."
        )
        if words:
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
            logger.info("Sending request to OpenRouter (%s)...", self.model_name)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content or ""
            invoice_number = self._clean_output(raw_text)

            latency = (time.time() - start_time) * 1000

            result = {
                "invoice_number": invoice_number,
                "raw_response": raw_text,
                "latency_ms": latency,
                "method": self.model_name,
            }
            if response.usage:
                result["usage"] = response.usage.model_dump()
            return result

        except Exception as exc:
            logger.error("OpenRouter inference failed: %s", exc)
            return {"invoice_number": None, "error": str(exc)}

    def _image_to_data_url(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _clean_output(self, text: str) -> Optional[str]:
        """Clean up JSON-first model output."""
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

        prefixes = ["invoice number:", "invoice #:", "inv #:", "no.", "invoice no."]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break

        return cleaned
