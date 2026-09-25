"""
FastAPI endpoints for Invoice NER API.
"""

import io
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

from . import inference
from .heuristics import extract_invoice_heuristics
from .postprocessing import postprocess_invoice_number
from .utils import normalize_boxes, parse_ocr_text_file
from .validation import validate_model_extraction

logger = logging.getLogger(__name__)

IMAGE_FILE = File(..., description="Invoice image file (JPG, PNG, etc.)")
OCR_FILE = File(..., description="OCR data file (TXT or JSON format)")


INFERENCE_REQUESTS = Counter(
    "inference_requests_total",
    "Total invoice extraction requests.",
    ["method", "status"],
)
INFERENCE_ERRORS = Counter(
    "inference_errors_total",
    "Total invoice extraction errors.",
    ["method"],
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "End-to-end invoice extraction request latency.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, float("inf")),
)
MODEL_INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Model-only inference latency for requests that fall through to the NER model.",
    ["backend", "model_name"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, float("inf")),
)
FALLBACK_TOTAL = Counter(
    "fallback_total",
    "Total requests that fell back from heuristics to model inference.",
)


# ============================================================================
# FASTAPI APP
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    inference.load_model()
    yield
    # Cleanup if needed
    print("🔄 Shutting down...")


app = FastAPI(
    title="Invoice NER API",
    description="Finetuned LayoutLMv3 model for extracting invoice numbers",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """Request model for predictions"""

    words: list[str]
    boxes: list[list[int]]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if inference.backend is not None else "unhealthy",
        "model_loaded": inference.backend is not None,
        "device": inference.DEVICE,
        "inference_backend": inference.INFERENCE_BACKEND,
        "triton_model_name": inference.TRITON_MODEL_NAME,
    }


@app.get("/metrics")
@app.get("/metrics/")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/runtime/config")
async def runtime_config():
    """Return non-secret runtime model serving configuration."""
    return {
        "inference_backend": inference.INFERENCE_BACKEND,
        "device": inference.DEVICE,
        "model_path": inference.MODEL_PATH,
        "base_model": inference.BASE_MODEL,
        "triton_url": inference.TRITON_URL,
        "triton_model_name": inference.TRITON_MODEL_NAME,
        "triton_model_version": inference.TRITON_MODEL_VERSION,
        "model_loaded": inference.backend is not None,
        "processor_loaded": inference.processor is not None,
    }


@app.post("/predict")
def predict(
    image: UploadFile = IMAGE_FILE,
    ocr_file: UploadFile = OCR_FILE,
):
    """
    Extract invoice number from an invoice image and OCR data

    Args:
        image: Invoice image file
        ocr_file: OCR data file in either:
            - Text format (.txt): x1,y1,x2,y2,x3,y3,x4,y4,text per line
            - JSON format (.json): {"words": [...], "bboxes": [...]}

    Returns:
        JSON with extracted invoice number, method used, and detailed predictions
    """
    start_time = time.perf_counter()
    extraction_method = "unknown"
    if inference.backend is None or inference.processor is None:
        INFERENCE_ERRORS.labels(method=extraction_method).inc()
        INFERENCE_REQUESTS.labels(method=extraction_method, status="error").inc()
        INFERENCE_LATENCY.observe(time.perf_counter() - start_time)
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and validate image
        image_bytes = image.file.read()
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image = pil_image.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e!s}")

        img_width, img_height = pil_image.size

        # Read and parse OCR file
        ocr_bytes = ocr_file.file.read()
        ocr_filename = ocr_file.filename.lower()

        try:
            if ocr_filename.endswith(".json"):
                # Parse JSON file
                ocr_data = json.loads(ocr_bytes.decode("utf-8"))
                words = ocr_data.get("words", [])
                boxes = ocr_data.get("bboxes", ocr_data.get("boxes", []))
                ocr_lines = ocr_data.get("ocr_lines", None)

                # Check if boxes need normalization
                needs_normalization = (
                    any(coord > 1000 for box in boxes for coord in box)
                    if boxes
                    else False
                )
                if needs_normalization:
                    logger.info(
                        "Detected pixel coordinates in JSON, normalizing to 0-1000 range"
                    )
                    boxes = normalize_boxes(boxes, img_width, img_height)

            elif ocr_filename.endswith(".txt"):
                # Parse text file - save to temp file for parse_ocr_text_file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as tmp:
                    tmp.write(ocr_bytes.decode("utf-8", errors="ignore"))
                    tmp_path = tmp.name

                try:
                    ocr_data = parse_ocr_text_file(tmp_path)
                    words = ocr_data["words"]
                    boxes = ocr_data["bboxes"]
                    ocr_lines = ocr_data.get("ocr_lines", None)

                    # Normalize boxes to 0-1000 range
                    boxes = normalize_boxes(boxes, img_width, img_height)
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)
            else:
                raise HTTPException(
                    status_code=400, detail="OCR file must be .txt or .json format"
                )

        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e!s}")
        except (KeyError, TypeError, UnicodeDecodeError, ValueError, OSError) as e:
            raise HTTPException(
                status_code=400, detail=f"Error parsing OCR file: {e!s}"
            )

        if not words or not boxes:
            raise HTTPException(
                status_code=400, detail="OCR file must contain valid words and bboxes"
            )

        logger.info("=" * 60)
        logger.info("API: Starting invoice extraction pipeline")
        logger.info(f"Total words: {len(words)}, Total boxes: {len(boxes)}")
        if ocr_lines:
            logger.info(f"OCR lines available: {len(ocr_lines)}")
        logger.info("=" * 60)

        # Step 1: Try heuristics first
        invoice_number, matched_indices = extract_invoice_heuristics(words, ocr_lines)

        if invoice_number:
            # Heuristic found a match
            extraction_method = "heuristic"
            logger.info(f"Using heuristic extraction result: '{invoice_number}'")
            labels = [
                "HEURISTIC_MATCH" if i in matched_indices else "LABEL_0"
                for i in range(len(words))
            ]
        else:
            # Step 2: Fall back to model
            extraction_method = "model"
            FALLBACK_TOTAL.inc()
            logger.info("🤖 Falling back to LayoutLMv3 model inference")
            model_start_time = time.perf_counter()
            result = inference.predict_invoice(pil_image, words, boxes)
            MODEL_INFERENCE_LATENCY.labels(
                backend=inference.INFERENCE_BACKEND,
                model_name=(
                    inference.TRITON_MODEL_NAME
                    if inference.INFERENCE_BACKEND == "triton"
                    else os.path.basename(inference.MODEL_PATH)
                ),
            ).observe(time.perf_counter() - model_start_time)
            invoice_number = result["invoice_number"]
            labels = result["labels"]
            logger.info(f"Model extracted: '{invoice_number}'")

            # Step 3: Apply postprocessing to model results
            if invoice_number:
                invoice_number = postprocess_invoice_number(invoice_number)

            # Step 4: Validate model extraction
            if invoice_number and not validate_model_extraction(invoice_number):
                if ";" in invoice_number:
                    logger.warning(
                        f"✗ Rejected model extraction '{invoice_number}': contains semicolon"
                    )
                else:
                    logger.warning(
                        f"✗ Rejected model extraction '{invoice_number}': no letters or numbers"
                    )
                invoice_number = None

        # Final result
        invoice_number = invoice_number or "Not Found"
        logger.info("=" * 60)
        logger.info(f"API RESULT: '{invoice_number}' (via {extraction_method})")
        logger.info("=" * 60)

        # Build detailed predictions
        predictions = []
        for i, (word, label) in enumerate(zip(words[: len(labels)], labels)):
            predictions.append(
                {
                    "word": word,
                    "label": label,
                    "is_invoice_number": label.startswith(("LABEL_1", "LABEL_2"))
                    or label == "HEURISTIC_MATCH",
                }
            )

        INFERENCE_REQUESTS.labels(method=extraction_method, status="success").inc()
        INFERENCE_LATENCY.observe(time.perf_counter() - start_time)

        return {
            "invoice_number": invoice_number,
            "extraction_method": extraction_method,
            "predictions": predictions,
            "total_words": len(words),
            "model_device": inference.DEVICE,
        }

    except HTTPException:
        INFERENCE_ERRORS.labels(method=extraction_method).inc()
        INFERENCE_REQUESTS.labels(method=extraction_method, status="error").inc()
        INFERENCE_LATENCY.observe(time.perf_counter() - start_time)
        raise
    except Exception as e:
        INFERENCE_ERRORS.labels(method=extraction_method).inc()
        INFERENCE_REQUESTS.labels(method=extraction_method, status="error").inc()
        INFERENCE_LATENCY.observe(time.perf_counter() - start_time)
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}")
