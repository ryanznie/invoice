"""
Model loading and inference for Invoice NER using ONNX Runtime or Triton Inference Server.
"""

import os
import logging
import threading
import numpy as np
from PIL import Image
from typing import List, Dict
from transformers import LayoutLMv3Processor
import onnxruntime as ort
from abc import ABC, abstractmethod
import tritonclient.http as httpclient
from .openrouter import OpenRouterClient

from .validation import validate_image, validate_words, validate_boxes

ort.set_default_logger_severity(3)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Backend Configuration
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "onnx").lower()
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "layoutlmv3-lora-invoice-number")
TRITON_MODEL_VERSION = os.getenv("TRITON_MODEL_VERSION", "1")

# Load configuration from environment variables with defaults
# Default to the quantized model if available, otherwise standard ONNX
DEFAULT_MODEL_PATH = "models/artifacts/layoutlmv3_invoice_ner.onnx"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/layoutlmv3-base")
PROCESSOR_PATH = os.getenv("PROCESSOR_PATH")
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
processor = None
backend = None
model = None  # Alias for backward compatibility (ONNX session)
openrouter_client = None

# ============================================================================
# BACKEND ABSTRACTION
# ============================================================================


class InferenceBackend(ABC):
    @abstractmethod
    def load(self, model_path: str):
        """Load the model or establish connection"""
        pass

    @abstractmethod
    def predict(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Run inference and return logits"""
        pass


class OnnxBackend(InferenceBackend):
    def __init__(self):
        self.session = None

    def load(self, model_path: str):
        print(f"🚀 Loading ONNX model from {model_path}...")
        print(f"📱 Using device: {DEVICE} (Providers: {PROVIDERS})")

        try:
            try:
                print(f"👉 Attempting to load with providers: {PROVIDERS}")
                self.session = ort.InferenceSession(model_path, providers=PROVIDERS)
            except Exception as e:
                if "CoreMLExecutionProvider" in PROVIDERS:
                    print(f"⚠️ Failed to load with CoreML: {e}")
                    print("🔄 Falling back to CPUExecutionProvider only...")
                    self.session = ort.InferenceSession(
                        model_path, providers=["CPUExecutionProvider"]
                    )
                else:
                    raise e

            print(
                f"⚙️ Final Config: Model={model_path}, Device={DEVICE}, Providers={self.session.get_providers()}"
            )
            print("✅ ONNX Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            raise

    def predict(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        if self.session is None:
            raise ValueError("ONNX Session not loaded.")

        try:
            outputs = self.session.run(None, inputs)
            return outputs[0]  # Return logits
        except Exception as e:
            logger.error(f"ONNX Inference failed: {e}")
            raise


class TritonBackend(InferenceBackend):
    def __init__(self):
        self.model_name = TRITON_MODEL_NAME
        self.model_version = TRITON_MODEL_VERSION
        self._thread_local = threading.local()

    def load(self, model_path: str):
        # We ignore model_path for Triton connection, but we can verify server health
        print(f"🚀 Connecting to Triton Server at {TRITON_URL}...")
        try:
            client = self._get_client()
            if not client.is_server_live():
                raise ConnectionError("Triton server is not live")
            if not client.is_server_ready():
                raise ConnectionError("Triton server is not ready")
            if not client.is_model_ready(self.model_name):
                raise ValueError(
                    f"Model {self.model_name} is not ready on Triton server"
                )

            print(f"✅ Connected to Triton Server! Model '{self.model_name}' is ready.")
        except Exception as e:
            print(f"❌ Failed to connect to Triton: {e}")
            raise

    def _get_client(self):
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
            self._thread_local.client = client
        return client

    def predict(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        try:
            client = self._get_client()
        except Exception as e:
            raise ValueError(f"Failed to create Triton client: {e}")

        # Prepare Triton inputs
        triton_inputs = []
        for name, data in inputs.items():
            # Map input names if necessary (LayoutLMv3 usually uses standard names)
            # data matches numpy array
            # Create InferInput
            # Helper to map numpy types to triton types string if needed,
            # but set_data_from_numpy handles it usually if type is standard.

            # Explicit type conversion might be safer since Triton follows strong typing
            triton_type = self._get_triton_datatype(data.dtype)
            infer_input = httpclient.InferInput(name, data.shape, triton_type)
            infer_input.set_data_from_numpy(data)
            triton_inputs.append(infer_input)

        try:
            response = client.infer(model_name=self.model_name, inputs=triton_inputs)
            # We assume the first output is logits, or look for 'logits' if available
            # If we don't specify outputs, it returns all.
            # Let's try to get 'logits' or fall back to the first available output

            # Note: response.as_numpy(name) requires output name.
            # We can inspect response.get_output(name) but we need names first.

            # Get model metadata to find output name if strictly needed,
            # but standard is often 'logits'. Let's trust the server returns what we need
            # or use the first output since we only expect one main output (logits).

            # with httpclient, response is a wrapper

            # response.get_response() is metadata JSON
            output_name = response.get_response()["outputs"][0]["name"]
            return response.as_numpy(output_name)

        except Exception as e:
            logger.error(f"Triton Inference failed: {e}")
            raise

    def _get_triton_datatype(self, numpy_dtype):
        # Simple mapper, extend as needed
        if numpy_dtype == np.int64:
            return "INT64"
        if numpy_dtype == np.int32:
            return "INT32"
        if numpy_dtype == np.float32:
            return "FP32"
        if numpy_dtype == np.float64:
            return "FP64"
        return "FP32"  # Fallback/Assumption


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_model():
    """Load the model (backend) and processor"""
    global processor, backend

    # Resolve paths
    model_path = MODEL_PATH
    processor_path = PROCESSOR_PATH or BASE_MODEL

    if os.path.isdir(model_path):
        logger.warning(
            f"MODEL_PATH {model_path} is a directory. Assuming it contains processor config."
        )
        processor_path = model_path
        potential_onnx = os.path.join(model_path, "model.onnx")
        if os.path.exists(potential_onnx):
            model_path = potential_onnx
    elif os.path.isfile(model_path) and not PROCESSOR_PATH:
        processor_path = os.path.dirname(model_path)
    elif not os.path.exists(model_path) and INFERENCE_BACKEND == "onnx":
        print(f"⚠️ MODEL_PATH {model_path} not found.")

    # Load processor
    try:
        processor = LayoutLMv3Processor.from_pretrained(processor_path, apply_ocr=False)
        print(f"✅ Processor loaded from {processor_path}")
    except Exception as e:
        print(f"⚠️ Could not load processor from {processor_path}: {e}")
        print(f"   Falling back to base model: {BASE_MODEL}")
        processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL, apply_ocr=False)

    # Initialize Backend
    print(f"👉 Initializing Inference Backend: {INFERENCE_BACKEND.upper()}")

    if INFERENCE_BACKEND == "triton":
        backend = TritonBackend()
    else:
        # Default to ONNX
        backend = OnnxBackend()

    # Load Backend (Connect or Load File)
    backend.load(model_path)

    # helper for backward compatibility
    if isinstance(backend, OnnxBackend):
        global model
        model = backend.session

    # Initialize OpenRouter Client for fallback
    global openrouter_client
    openrouter_client = OpenRouterClient()
    # We do a lazy load in predict, but we can verify API key here if needed
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.warning(
            "OPENROUTER_API_KEY not found. OpenRouter fallback will be disabled."
        )


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================


def predict_invoice(
    image: Image.Image, words: List[str], boxes: List[List[int]]
) -> Dict:
    """
    Run inference on invoice using the configured backend
    """
    # Validate model loaded
    if backend is None or processor is None:
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
        return_tensors="np",
    )

    # Prepare inputs dictionary
    inputs = {
        "pixel_values": encoding["pixel_values"],
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "bbox": encoding["bbox"],
    }

    # Inference via Backend
    try:
        logits = backend.predict(inputs)
    except Exception as e:
        logger.error(f"Primary model failed: {e}")
        logger.info("🔄 Attempting fallback to OpenRouter...")

        if openrouter_client:
            openrouter_result = openrouter_client.predict(image=image, words=words)

            if openrouter_result.get("error"):
                logger.error(
                    "OpenRouter fallback also failed: %s",
                    openrouter_result["error"],
                )
                raise e  # Re-raise original error if fallback fails

            # Construct result compatible with existing pipeline
            if openrouter_result["invoice_number"]:
                # Create labels with HEURISTIC_MATCH style (or just generic)
                # Since we don't have token-level predictions, we'll mark all as LABEL_0
                # effectively bypassing the token visualization for the fallback result
                # but returning the correct extracted value.
                invoice_number = openrouter_result["invoice_number"]
                logger.info(f"OpenRouter extracted: {invoice_number}")

                return {
                    "words": words,
                    "labels": ["LABEL_0"] * len(words),  # Dummy labels
                    "invoice_number": invoice_number,
                    "confidence_scores": [0.0] * len(words),
                    "method": openrouter_result.get("method", "openrouter"),
                }
            else:
                # OpenRouter returned nothing
                logger.warning("OpenRouter found no invoice number.")
                raise e
        else:
            raise e

    # Post-processing
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

    # Map IDs to Labels
    # Use processor's id2label if available, else default
    if hasattr(processor, "id2label") and processor.id2label:
        id2label = processor.id2label
    else:
        id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}

    token_boxes = encoding["bbox"][0].tolist()

    prev_word_idx = None

    # Iterate through predictions
    # Note: predictions[0], confidence_scores[0] because batch size is 1
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
            # Handle integer keys in id2label (JSON keys are strings sometimes in configs)
            # but here internal dict usually int keys.
            label = id2label.get(pred, id2label.get(str(pred), "LABEL_0"))

            # Only take first subtoken prediction for each word
            if word_idx != prev_word_idx:
                predicted_labels.append(label)
                word_confidences.append(conf)

                # Extract invoice number tokens (Assuming standard labels or user specific)
                # Adjust these labels if your model uses different names (e.g., B-INVOICE, I-INVOICE)
                # The user previous code had explicit LABEL_1/LABEL_2 checks, so we keep that logic
                # or match user intent. The previous code usage:
                # if label == "LABEL_1" or label == "LABEL_2":

                # Check for "INVOICE" strings just in case logic changes,
                # but let's stick to the previous implementation's specific logic:
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
