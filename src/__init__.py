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

# Gradio is a development UI and is intentionally absent from the production worker.
try:
    from .gradio_ui import create_gradio_interface, gradio_predict, create_annotated_image
except ModuleNotFoundError as exc:
    if exc.name != "gradio":
        raise

    def _gradio_unavailable(*args, **kwargs):
        raise RuntimeError("Gradio is not installed in this runtime")

    create_gradio_interface = _gradio_unavailable
    gradio_predict = _gradio_unavailable
    create_annotated_image = _gradio_unavailable

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
    # Gradio
    "create_gradio_interface",
    "gradio_predict",
    "create_annotated_image",
]
