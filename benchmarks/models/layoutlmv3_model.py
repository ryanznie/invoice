"""
LayoutLMv3 model implementation with LoRA adapters.

This module directly loads your fine-tuned LayoutLMv3-LoRA model
for benchmarking.
"""

import os
import sys
from typing import Dict, Optional, List, Any
from PIL import Image
import logging
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from peft import PeftModel

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from benchmarks.models.base import BaseInvoiceModel, InferenceResult

logger = logging.getLogger(__name__)


class LayoutLMv3Model(BaseInvoiceModel):
    """
    LayoutLMv3 with LoRA fine-tuning for invoice number extraction.

    This model loads your fine-tuned LoRA adapters and uses multimodal
    understanding (image + text + layout) to extract invoice numbers.
    """

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize LayoutLMv3 model.

        Args:
            model_config: Configuration dictionary with optional keys:
                - model_path: Path to LoRA adapter weights
                - base_model: HuggingFace base model identifier
                - device: 'cpu', 'cuda', or 'mps'
                - max_length: Maximum sequence length
                - num_labels: Number of NER labels (default: 3)
        """
        super().__init__(model_config)
        self.model = None
        self.processor = None

        # Set configuration with defaults
        self.model_path = self.model_config.get(
            "model_path", "models/layoutlmv3-lora-invoice-number"
        )
        self.base_model = self.model_config.get(
            "base_model", "microsoft/layoutlmv3-base"
        )
        self.device = self.model_config.get("device", "cpu")
        self.max_length = self.model_config.get("max_length", 512)
        self.num_labels = self.model_config.get("num_labels", 3)

    def load(self) -> None:
        """
        Load the LayoutLMv3 model with your fine-tuned LoRA adapters.

        This directly loads the model without relying on src.inference globals.
        """
        try:
            logger.info(f"🚀 Loading LayoutLMv3-LoRA from {self.model_path}")
            logger.info(f"📱 Using device: {self.device}")

            # Load processor
            self.processor = LayoutLMv3Processor.from_pretrained(
                self.model_path, apply_ocr=False
            )
            logger.info("✓ Processor loaded")

            # Load base model
            base_model = LayoutLMv3ForTokenClassification.from_pretrained(
                self.base_model, num_labels=self.num_labels
            )
            logger.info(f"✓ Base model loaded: {self.base_model}")

            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            logger.info(f"✓ LoRA adapters loaded from: {self.model_path}")

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✓ Model moved to {self.device} and set to eval mode")

            logger.info("✅ LayoutLMv3 model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load LayoutLMv3 model: {e}")
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
        Run LayoutLMv3 inference on invoice data.

        Args:
            image: PIL Image of the invoice
            words: List of OCR words
            boxes: List of bounding boxes [x0, y0, x1, y1] normalized to 0-1000
            text: Not used (included for interface compatibility)
            **kwargs: Additional parameters

        Returns:
            InferenceResult with extracted invoice number and metadata

        Raises:
            ValueError: If required inputs (image, words, boxes) are missing
        """
        if image is None or words is None or boxes is None:
            raise ValueError(
                "LayoutLMv3 requires image, words, and boxes for inference"
            )

        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load() first.")

        try:
            # Process inputs with the processor
            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Get word_ids before moving to device
            word_ids = encoding.word_ids(0)

            # Move to device
            encoding_device = {k: v.to(self.device) for k, v in encoding.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**encoding_device)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2)

                # Get confidence scores
                probs = torch.softmax(logits, dim=2)
                confidence_scores = torch.max(probs, dim=2).values

            # Decode predictions
            predicted_labels = []
            invoice_tokens = []
            word_confidences = []

            token_boxes = encoding["bbox"][0].tolist()

            prev_word_idx = None
            for idx, (pred, box, word_idx, conf) in enumerate(
                zip(
                    predictions[0].cpu().tolist(),
                    token_boxes,
                    word_ids,
                    confidence_scores[0].cpu().tolist(),
                )
            ):
                # Skip special tokens and padding
                if box != [0, 0, 0, 0] and word_idx is not None:
                    label = self.model.config.id2label[pred]

                    # Only take first subtoken prediction for each word
                    if word_idx != prev_word_idx:
                        predicted_labels.append(label)
                        word_confidences.append(conf)

                        # Extract invoice number tokens (LABEL_1 = B-INVOICE, LABEL_2 = I-INVOICE)
                        if label.startswith("LABEL_1") or label.startswith("LABEL_2"):
                            invoice_tokens.append(words[word_idx])

                        prev_word_idx = word_idx

            # Combine invoice tokens
            invoice_number = " ".join(invoice_tokens) if invoice_tokens else None

            # Calculate average confidence
            avg_confidence = None
            if word_confidences:
                avg_confidence = sum(word_confidences) / len(word_confidences)

            return InferenceResult(
                invoice_number=invoice_number,
                confidence=avg_confidence,
                method="model",
                metadata={
                    "labels": predicted_labels,
                    "total_words": len(predicted_labels),
                    "device": self.device,
                    "num_invoice_tokens": len(invoice_tokens),
                },
            )

        except Exception as e:
            logger.error(f"LayoutLMv3 prediction failed: {e}")
            return InferenceResult(
                invoice_number=None,
                confidence=None,
                method="model",
                metadata={"error": str(e)},
            )

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for logging."""
        return {
            "model_name": "LayoutLMv3-LoRA",
            "model_version": "v1.0",
            "checkpoint_path": self.model_path,
            "base_model": self.base_model,
            "device": self.device,
            "max_length": self.max_length,
            "architecture": "LayoutLMv3 + LoRA adapters",
        }

    def cleanup(self) -> None:
        """Release model resources."""
        import torch

        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # Clear CUDA cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("LayoutLMv3 model resources released")
