"""
Model loading and inference for Invoice NER using ONNX Runtime.
"""

import os
import logging
import numpy as np
from PIL import Image
from typing import List, Dict
from transformers import LayoutLMv3Processor
import onnxruntime as ort
from .validation import validate_image, validate_words, validate_boxes

ort.set_default_logger_severity(3)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load configuration from environment variables with defaults
# Default to the quantized model if available, otherwise standard ONNX
DEFAULT_MODEL_PATH = "models/artifacts/layoutlmv3_invoice_ner.onnx"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/layoutlmv3-base")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
NUM_LABELS = int(os.getenv("NUM_LABELS", "3"))

# Device selection: environment variable > MPS > CPU
# Note: ONNX Runtime providers need to be configured explicitly
device_env = os.getenv("DEVICE", "").lower()
PROVIDERS = ["CPUExecutionProvider"]
if device_env == "cuda" or (device_env == "" and ort.get_device() == "GPU"):
    if "CUDAExecutionProvider" in ort.get_available_providers():
        PROVIDERS.insert(0, "CUDAExecutionProvider")
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
elif device_env == "mps":
    if "CoreMLExecutionProvider" in ort.get_available_providers():
        PROVIDERS.insert(0, "CoreMLExecutionProvider")
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
else:
    DEVICE = "cpu"

logger.debug(f"Using device: {DEVICE} with providers: {PROVIDERS}")

# Global model session and processor
session = None
model = None  # Alias for backward compatibility
processor = None


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_model():
    """Load the LayoutLMv3 ONNX model and processor"""
    global session, processor, model

    # Resolve paths
    # If MODEL_PATH is a directory (legacy config), we use it for processor
    # but need to find the ONNX file elsewhere
    model_path = MODEL_PATH

    if os.path.isdir(model_path):
        logger.warning(
            f"MODEL_PATH {model_path} is a directory. Assuming it contains processor config."
        )
        processor_path = model_path

        # Check for standard model.onnx inside the directory
        potential_onnx = os.path.join(model_path, "model.onnx")
        if os.path.exists(potential_onnx):
            logger.info(f"Found ONNX model at: {potential_onnx}")
            model_path = potential_onnx
        else:
            raise ValueError(
                f"MODEL_PATH is a directory ({model_path}) but 'model.onnx' was not found inside it. Please point MODEL_PATH to the .onnx file directly."
            )

    elif os.path.isfile(model_path):
        # It's a file (likely the .onnx file)
        # Use its parent directory for processor config
        processor_path = os.path.dirname(model_path)

    elif not os.path.exists(model_path):
        # Path doesn't exist, will fail later but set defaults for now
        processor_path = BASE_MODEL
        raise ValueError(f"MODEL_PATH {model_path} not found.")

    print(f"🚀 Loading ONNX model from {model_path}...")
    print(f"📱 Using device: {DEVICE} (Providers: {PROVIDERS})")

    # Load processor
    try:
        # Try loading from the resolved processor path
        processor = LayoutLMv3Processor.from_pretrained(processor_path, apply_ocr=False)
        print(f"✅ Processor loaded from {processor_path}")
    except Exception as e:
        print(f"⚠️ Could not load processor from {processor_path}: {e}")
        print(f"   Falling back to base model: {BASE_MODEL}")
        processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL, apply_ocr=False)

    # Create ONNX Runtime session
    try:
        try:
            print(f"👉 Attempting to load with providers: {PROVIDERS}")
            session = ort.InferenceSession(model_path, providers=PROVIDERS)
        except Exception as e:
            if "CoreMLExecutionProvider" in PROVIDERS:
                print(f"⚠️ Failed to load with CoreML: {e}")
                print("🔄 Falling back to CPUExecutionProvider only...")
                session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
            else:
                raise e

        model = session  # Alias for backward compatibility
        print(
            f"⚙️ Final Config: Model={model_path}, Device={DEVICE}, Providers={session.get_providers()}"
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        raise


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================


def predict_invoice(
    image: Image.Image, words: List[str], boxes: List[List[int]]
) -> Dict:
    """
    Run inference on invoice using ONNX Runtime

    Args:
        image: PIL Image
        words: List of OCR words
        boxes: List of bounding boxes [x0, y0, x1, y1] normalized to 0-1000

    Returns:
        Dictionary with predictions and invoice number

    Raises:
        ValueError: If model not loaded, inputs are invalid, or dimensions mismatch
    """
    # Validate model loaded
    if session is None or processor is None:
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
        return_tensors="np",  # Return numpy arrays for ONNX
    )

    # Prepare inputs for ONNX Runtime
    input_feed = {
        "pixel_values": encoding["pixel_values"],
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "bbox": encoding["bbox"],
    }

    # Inference
    try:
        outputs = session.run(None, input_feed)
        logits = outputs[0]  # Logits are the first output
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

    # Post-processing (similar to PyTorch version but using numpy)
    predictions = np.argmax(logits, axis=2)

    # Get confidence scores
    # Softmax on last axis (axis 2)
    exp_logits = np.exp(logits - np.max(logits, axis=2, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=2, keepdims=True)
    confidence_scores = np.max(probs, axis=2)

    # Get word ids
    word_ids = encoding.word_ids(0)

    # Decode predictions
    predicted_labels = []
    invoice_tokens = []
    word_confidences = []

    # Get ID to Label mapping from model config if available, otherwise use default
    # The processor should have the label map if loaded from fine-tuned checkpoint,
    # but for safety we define default mapping for this specific task
    # User requested to keep original labels (LABEL_0, LABEL_1, LABEL_2)
    id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}

    token_boxes = encoding["bbox"][0].tolist()

    prev_word_idx = None
    for idx, (pred, box, word_idx, conf) in enumerate(
        zip(
            predictions[0].tolist(),
            token_boxes,
            word_ids,
            confidence_scores[0].tolist(),
        )
    ):
        # Skip special tokens and padding
        if box != [0, 0, 0, 0] and word_idx is not None:
            label = id2label.get(pred, "LABEL_0")

            # Only take first subtoken prediction for each word
            if word_idx != prev_word_idx:
                predicted_labels.append(label)
                word_confidences.append(conf)

                # Extract invoice number tokens
                if label == "LABEL_1" or label == "LABEL_2":
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
