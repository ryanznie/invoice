"""
Hybrid model combining heuristics and ML model.

This implements the production inference pipeline:
1. Try heuristic extraction first (fast)
2. If heuristic fails, fall back to ML model (slower but more robust)
"""

import os
import sys
import logging
from typing import Dict, Optional, List, Any
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from benchmarks.models.base import BaseInvoiceModel, InferenceResult

logger = logging.getLogger(__name__)


class HybridModel(BaseInvoiceModel):
    """
    Hybrid inference pipeline: Heuristics → Model fallback.

    This is the recommended production approach:
    - Fast: Heuristics succeed ~70% of the time with near-zero latency
    - Robust: Model fallback handles cases where heuristics fail
    - Efficient: Only pays ML inference cost when necessary

    The benchmarking script will track:
    - Fallback rate (how often model is used)
    - Average latency for heuristic-only vs hybrid cases
    - Accuracy of each component
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        fallback_model: Optional[BaseInvoiceModel] = None,
    ):
        """
        Initialize hybrid model.

        Args:
            model_config: Configuration dictionary
            fallback_model: The ML model to use when heuristics fail.
                           If None, uses LayoutLMv3Model by default.
        """
        super().__init__(model_config)

        self.fallback_model = fallback_model
        self.heuristic_success_count = 0
        self.fallback_count = 0

    def load(self) -> None:
        """
        Load the fallback model.

        Note: Heuristics don't require loading, they're pure pattern matching.
        """
        if self.fallback_model is None:
            # Default to LayoutLMv3
            from benchmarks.models.layoutlmv3_model import LayoutLMv3Model

            self.fallback_model = LayoutLMv3Model(self.model_config)

        logger.info("Loading hybrid model...")
        logger.info("✓ Heuristics ready (no loading required)")

        # Load the fallback ML model
        logger.info("Loading fallback model...")
        self.fallback_model.load()

        logger.info("✅ Hybrid model loaded successfully")

    def predict(
        self,
        image: Optional[Image.Image] = None,
        words: Optional[List[str]] = None,
        boxes: Optional[List[List[int]]] = None,
        text: Optional[str] = None,
        **kwargs,
    ) -> InferenceResult:
        """
        Run hybrid inference: heuristics first, then model if needed.

        Args:
            image: PIL Image of the invoice
            words: List of OCR words
            boxes: List of bounding boxes [x0, y0, x1, y1]
            text: Plain text content (optional, for heuristics)
            **kwargs: Additional parameters

        Returns:
            InferenceResult with extraction method indicated
        """
        if words is None:
            raise ValueError("Hybrid model requires at least 'words' for inference")

        # STEP 1: Try heuristics first
        try:
            from src.heuristics import extract_invoice_heuristics

            # Extract using heuristics
            invoice_number, matched_indices = extract_invoice_heuristics(
                words=words, ocr_lines=kwargs.get("ocr_lines")
            )

            if invoice_number:
                # Heuristic success!
                self.heuristic_success_count += 1
                logger.debug(f"✓ Heuristic extraction successful: {invoice_number}")

                return InferenceResult(
                    invoice_number=invoice_number,
                    confidence=1.0,  # Heuristics are deterministic when they match
                    method="heuristic",
                    metadata={
                        "matched_indices": matched_indices,
                        "fallback_used": False,
                        "extraction_stage": "heuristic",
                    },
                )

        except Exception as e:
            logger.warning(f"Heuristic extraction failed: {e}")

        # STEP 2: Fallback to ML model
        logger.debug("Heuristics failed, falling back to ML model...")
        self.fallback_count += 1

        try:
            result = self.fallback_model.predict(
                image=image, words=words, boxes=boxes, text=text, **kwargs
            )

            # Update metadata to indicate fallback was used
            if result.metadata is None:
                result.metadata = {}

            result.metadata["fallback_used"] = True
            result.metadata["extraction_stage"] = "model_fallback"
            result.method = "model_fallback"

            return result

        except Exception as e:
            logger.error(f"Model fallback also failed: {e}")

            return InferenceResult(
                invoice_number=None,
                confidence=None,
                method="failed",
                metadata={
                    "fallback_used": True,
                    "extraction_stage": "failed",
                    "error": str(e),
                },
            )

    def get_config(self) -> Dict[str, Any]:
        """Return hybrid model configuration."""
        fallback_config = {}
        if self.fallback_model:
            fallback_config = self.fallback_model.get_config()

        return {
            "model_name": "HybridModel (Heuristics + ML)",
            "model_version": "v1.0",
            "architecture": "Heuristics → Model Fallback",
            "heuristic_patterns": 14,  # From heuristics.py
            "fallback_model": fallback_config,
            "stats": {
                "heuristic_success_count": self.heuristic_success_count,
                "fallback_count": self.fallback_count,
            },
        }

    def cleanup(self) -> None:
        """Release fallback model resources."""
        if self.fallback_model:
            self.fallback_model.cleanup()

        logger.info(
            f"Hybrid model cleanup complete. "
            f"Stats: {self.heuristic_success_count} heuristic successes, "
            f"{self.fallback_count} fallbacks"
        )
