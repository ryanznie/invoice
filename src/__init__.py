"""
Invoice NER package - Modular components for invoice number extraction.
"""

# Configuration and model
from .inference import (
    load_model,
    predict_invoice,
    MODEL_PATH,
    BASE_MODEL,
    MAX_LENGTH,
    NUM_LABELS,
    DEVICE,
    model,
    processor,
)

# Heuristics and postprocessing
from .heuristics import extract_invoice_heuristics
from .postprocessing import postprocess_invoice_number

# Validation
from .validation import (
    validate_image,
    validate_words,
    validate_boxes,
    validate_model_extraction,
)

# Utilities
from .utils import parse_ocr_text_file, normalize_boxes

# API
from .api import app

# Visualization
from .visualization import create_annotated_image

__all__ = [
    # Inference
    "load_model",
    "predict_invoice",
    "MODEL_PATH",
    "BASE_MODEL",
    "MAX_LENGTH",
    "NUM_LABELS",
    "DEVICE",
    "model",
    "processor",
    # Heuristics & Postprocessing
    "extract_invoice_heuristics",
    "postprocess_invoice_number",
    # Validation
    "validate_image",
    "validate_words",
    "validate_boxes",
    "validate_model_extraction",
    # Utils
    "parse_ocr_text_file",
    "normalize_boxes",
    # API
    "app",
    # Visualization
    "create_annotated_image",
]
