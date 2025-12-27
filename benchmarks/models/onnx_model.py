"""
ONNX Runtime implementation for invoice extraction models.
"""

import os
import sys
import logging
import numpy as np
import onnxruntime as ort
from typing import Dict, Optional, List, Any
from PIL import Image
from transformers import LayoutLMv3Processor

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from benchmarks.models.base import BaseInvoiceModel, InferenceResult

logger = logging.getLogger(__name__)


class OnnxModel(BaseInvoiceModel):
    """
    LayoutLMv3 model using ONNX Runtime for inference.
    """

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize ONNX model wrapper.

        Args:
            model_config: Configuration dictionary with optional keys:
                - model_path: Path to .onnx file
                - base_model: HuggingFace base model identifier (for processor)
                - providers: List of execution providers (default: CPU)
                - max_length: Maximum sequence length (default: 512)
        """
        super().__init__(model_config)
        self.session = None
        self.processor = None

        # Configuration
        self.model_path = (
            self.model_config.get("model_path")
            or "models/artifacts/layoutlmv3_invoice_ner.onnx"
        )
        self.base_model = self.model_config.get(
            "base_model", "microsoft/layoutlmv3-base"
        )
        self.max_length = int(self.model_config.get("max_length", 512))

        # Configure providers
        self.providers = self.model_config.get("providers", ["CPUExecutionProvider"])
        if not self.providers:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "CoreMLExecutionProvider" in ort.get_available_providers():
                self.providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                self.providers = ["CPUExecutionProvider"]

    def load(self) -> None:
        """
        Load ONNX session and processor.
        """
        logger.info(f"Loading ONNX model from {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found at {self.model_path}")

        # Load processor
        try:
            # First try loading processor from model directory (if it contains config)
            # or use base model
            processor_path = os.path.dirname(self.model_path)
            try:
                self.processor = LayoutLMv3Processor.from_pretrained(
                    processor_path, apply_ocr=False
                )
                logger.info(f"✓ Processor loaded from {processor_path}")
            except Exception:
                logger.info(
                    f"Could not load processor from {processor_path}, using base model"
                )
                self.processor = LayoutLMv3Processor.from_pretrained(
                    self.base_model, apply_ocr=False
                )
                logger.info(f"✓ Processor loaded from {self.base_model}")

        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise

        # Load ONNX session
        try:
            logger.info(f"Creating inference session with providers: {self.providers}")
            self.session = ort.InferenceSession(
                self.model_path, providers=self.providers
            )
            logger.info("✅ ONNX session initialized")
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
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
        Run ONNX inference.
        """
        if self.session is None or self.processor is None:
            raise ValueError("Model not loaded. Call load() first.")

        if image is None or words is None or boxes is None:
            raise ValueError("OnnxModel requires image, words, and boxes")

        try:
            # Process inputs
            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="np",
            )

            # Prepare ONNX inputs
            input_feed = {
                "pixel_values": encoding["pixel_values"],
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "bbox": encoding["bbox"],
            }

            # Run inference
            outputs = self.session.run(None, input_feed)
            logits = outputs[0]

            # Post-processing
            predictions = np.argmax(logits, axis=2)

            # Confidence scores
            exp_logits = np.exp(logits - np.max(logits, axis=2, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=2, keepdims=True)
            confidence_scores = np.max(probs, axis=2)

            # Decode
            predicted_labels = []
            invoice_tokens = []
            word_confidences = []

            # Label mapping (defaulting to standard BIO scheme for 3 labels)
            # 0: O, 1: B-INVOICE, 2: I-INVOICE
            id2label = {0: "O", 1: "B-INVOICE", 2: "I-INVOICE"}

            word_ids = encoding.word_ids(0)
            token_boxes = encoding["bbox"][0].tolist()

            prev_word_idx = None

            for pred, box, word_idx, conf in zip(
                predictions[0].tolist(),
                token_boxes,
                word_ids,
                confidence_scores[0].tolist(),
            ):
                if box != [0, 0, 0, 0] and word_idx is not None:
                    label = id2label.get(pred, "O")

                    if word_idx != prev_word_idx:
                        predicted_labels.append(label)
                        word_confidences.append(conf)

                        if label in ["B-INVOICE", "I-INVOICE"]:
                            invoice_tokens.append(words[word_idx])

                        prev_word_idx = word_idx

            invoice_number = " ".join(invoice_tokens) if invoice_tokens else None

            avg_confidence = None
            if word_confidences:
                avg_confidence = sum(word_confidences) / len(word_confidences)

            return InferenceResult(
                invoice_number=invoice_number,
                confidence=avg_confidence,
                method="onnx_model",
                metadata={
                    "model_path": self.model_path,
                    "providers": self.session.get_providers(),
                    "total_tokens": len(predicted_labels),
                },
            )

        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return InferenceResult(
                invoice_number=None,
                confidence=None,
                method="failed",
                metadata={"error": str(e)},
            )

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": "LayoutLMv3 (ONNX)",
            "model_version": "v1.0",
            "model_path": self.model_path,
            "providers": (
                self.session.get_providers() if self.session else self.providers
            ),
            "max_length": self.max_length,
        }
