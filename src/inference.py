"""
Model loading and inference for Invoice NER.
"""

import os
import logging
import torch
from PIL import Image
from typing import List, Dict
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from peft import PeftModel
from .validation import validate_image, validate_words, validate_boxes

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load configuration from environment variables with defaults
MODEL_PATH = os.getenv("MODEL_PATH", "models/layoutlmv3-lora-invoice-number")
BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/layoutlmv3-base")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
NUM_LABELS = int(os.getenv("NUM_LABELS", "3"))

# Device selection: environment variable > MPS > CPU
device_env = os.getenv("DEVICE", "").lower()
if device_env in ["cpu", "cuda", "mps"]:
    DEVICE = device_env
    logger.debug(f"Using device from DEVICE env var: {DEVICE}")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    logger.debug("DEVICE env var not set, auto-detected MPS")
else:
    DEVICE = "cpu"
    logger.debug("DEVICE env var not set, defaulting to CPU")

# Global model and processor
model = None
processor = None


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_model():
    """Load the LayoutLMv3 model with LoRA adapters"""
    global model, processor

    print(f"🚀 Loading model from {MODEL_PATH}...")
    print(f"📱 Using device: {DEVICE}")

    # Load processor
    processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH, apply_ocr=False)

    # Load base model + LoRA adapter
    base = LayoutLMv3ForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS
    )
    model = PeftModel.from_pretrained(base, MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    print("✅ Model loaded successfully!")


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================


def predict_invoice(
    image: Image.Image, words: List[str], boxes: List[List[int]]
) -> Dict:
    """
    Run inference on invoice

    Args:
        image: PIL Image
        words: List of OCR words
        boxes: List of bounding boxes [x0, y0, x1, y1] normalized to 0-1000

    Returns:
        Dictionary with predictions and invoice number

    Raises:
        ValueError: If model not loaded, inputs are invalid, or dimensions mismatch
        TypeError: If inputs have incorrect types
    """
    # Validate model loaded
    if model is None or processor is None:
        raise ValueError("Model not loaded. Call load_model() first.")

    # Validate inputs
    validate_image(image)
    validate_words(words)
    validate_boxes(boxes, words)

    # Process inputs
    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Get word_ids before moving to device
    word_ids = encoding.word_ids(0)

    # Move to device
    encoding_device = {k: v.to(DEVICE) for k, v in encoding.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**encoding_device)
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
            label = model.config.id2label[pred]

            # Only take first subtoken prediction for each word
            if word_idx != prev_word_idx:
                predicted_labels.append(label)
                word_confidences.append(conf)

                # Extract invoice number tokens
                if label.startswith("LABEL_1") or label.startswith("LABEL_2"):
                    invoice_tokens.append(words[word_idx])

                prev_word_idx = word_idx

    # Combine invoice tokens
    invoice_number = " ".join(invoice_tokens) if invoice_tokens else None

    return {
        "words": words[: len(predicted_labels)],
        "labels": predicted_labels,
        "invoice_number": invoice_number,
        "confidence_scores": word_confidences,
    }
