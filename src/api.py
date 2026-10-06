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
from fastapi.middleware.cors import CORSMiddleware
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
from .openrouter import OpenRouterConfigurationError
from .postprocessing import postprocess_invoice_number
from .utils import normalize_boxes, parse_ocr_text_file
from .validation import validate_boxes, validate_model_extraction, validate_words

logger = logging.getLogger(__name__)

IMAGE_FILE = File(..., description="Invoice image file (JPG, PNG, etc.)")
OCR_FILE = File(None, description="OCR data file (TXT or JSON format)")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
MAX_OCR_BYTES = int(os.getenv("MAX_OCR_BYTES", str(2 * 1024 * 1024)))


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
    """Load the inference model when the API starts."""
    inference.load_model()
    yield
    print("🔄 Shutting down...")


app = FastAPI(
    title="Invoice NER API",
    description="Finetuned LayoutLMv3 model for extracting invoice numbers",
    lifespan=lifespan,
)


def _get_cors_origins() -> list[str]:
    raw_origins = os.getenv("CORS_ORIGINS", "")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


cors_origins = _get_cors_origins()
cors_origin_regex = os.getenv("CORS_ORIGIN_REGEX")
if cors_origins or cors_origin_regex:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=cors_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class PredictionRequest(BaseModel):
    """Request model for predictions"""

    words: list[str]
    boxes: list[list[int]]


@app.get("/")
async def root():
    """API root metadata."""
    return {
        "name": "Invoice NER API",
        "status": "ok",
        "docs_url": "/docs",
        "predict_url": "/predict",
        "health_url": "/health",
    }


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


@app.get("/ping")
async def runpod_health_check():
    """Runpod load-balancer compatible readiness endpoint."""
    if inference.backend is None or inference.processor is None:
        return Response(status_code=204)
    return {"status": "healthy"}


@app.get("/metrics")
@app.get("/metrics/")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _parse_ocr_upload(ocr_file, image_width: int, image_height: int):
    """Return parsed OCR when valid; otherwise select image-only inference."""
    if ocr_file is None:
        return None

    try:
        ocr_bytes = ocr_file.file.read(MAX_OCR_BYTES + 1)
        if len(ocr_bytes) > MAX_OCR_BYTES:
            logger.info("Ignoring OCR file over the configured size limit")
            return None

        filename = (ocr_file.filename or "").lower()
        if filename.endswith(".json"):
            ocr_data = json.loads(ocr_bytes.decode("utf-8"))
            if not isinstance(ocr_data, dict):
                return None
            words = ocr_data.get("words")
            boxes = ocr_data.get("bboxes", ocr_data.get("boxes"))
            ocr_lines = ocr_data.get("ocr_lines")
            needs_normalization = (
                any(coord > 1000 for box in boxes for coord in box) if boxes else False
            )
            if needs_normalization:
                boxes = normalize_boxes(boxes, image_width, image_height)
        elif filename.endswith(".txt"):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(ocr_bytes.decode("utf-8"))
                tmp_path = tmp.name
            try:
                ocr_data = parse_ocr_text_file(tmp_path)
                words = ocr_data["words"]
                boxes = normalize_boxes(
                    ocr_data["bboxes"], image_width, image_height
                )
                ocr_lines = ocr_data.get("ocr_lines")
            finally:
                os.unlink(tmp_path)
        else:
            return None

        if (
            not isinstance(words, list)
            or not words
            or not all(isinstance(word, str) and word.strip() for word in words)
            or not isinstance(boxes, list)
            or not boxes
            or (
                ocr_lines is not None
                and (
                    not isinstance(ocr_lines, list)
                    or not all(isinstance(line, str) for line in ocr_lines)
                )
            )
        ):
            return None

        validate_words(words)
        validate_boxes(boxes, words)
        return words, boxes, ocr_lines
    except Exception as exc:
        logger.info(
            "Ignoring OCR file that could not be parsed (%s)", type(exc).__name__
        )
        return None


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
    ocr_file: UploadFile | None = OCR_FILE,
):
    """
    Extract an invoice number from an image, using valid OCR data when provided.

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
    try:
        # Read and validate image
        image_bytes = image.file.read(MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit",
            )
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image = pil_image.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e!s}")

        img_width, img_height = pil_image.size

        ocr_data = _parse_ocr_upload(ocr_file, img_width, img_height)
        if ocr_data is None:
            extraction_method = "openrouter_image_only"
            FALLBACK_TOTAL.inc()
            logger.info("No valid OCR data; using image-only vision inference")
            try:
                image_result = inference.predict_invoice_from_image(pil_image)
            except OpenRouterConfigurationError:
                raise
            except Exception as exc:
                logger.warning(
                    "Image-only vision inference failed (%s)", type(exc).__name__
                )
                raise HTTPException(
                    status_code=502, detail="Image-only invoice extraction failed."
                ) from exc

            invoice_number = image_result.get("invoice_number")
            if invoice_number:
                invoice_number = postprocess_invoice_number(invoice_number)
                if not validate_model_extraction(invoice_number):
                    invoice_number = None
            INFERENCE_REQUESTS.labels(method=extraction_method, status="success").inc()
            INFERENCE_LATENCY.observe(time.perf_counter() - start_time)
            return {
                "invoice_number": invoice_number or "Not Found",
                "extraction_method": extraction_method,
                "predictions": [],
                "total_words": 0,
                "model_device": inference.DEVICE,
            }

        words, boxes, ocr_lines = ocr_data
        if inference.backend is None or inference.processor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

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
            logger.info("Using heuristic extraction result")
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
            logger.info("Model extraction completed")

            # Step 3: Apply postprocessing to model results
            if invoice_number:
                invoice_number = postprocess_invoice_number(invoice_number)

            # Step 4: Validate model extraction
            if invoice_number and not validate_model_extraction(invoice_number):
                if ";" in invoice_number:
                    logger.warning(
                        "Rejected model extraction because it contains a semicolon"
                    )
                else:
                    logger.warning(
                        "Rejected model extraction because it has no letters or numbers"
                    )
                invoice_number = None

        # Final result
        invoice_number = invoice_number or "Not Found"
        logger.info("=" * 60)
        logger.info("API extraction completed via %s", extraction_method)
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
    except OpenRouterConfigurationError as e:
        INFERENCE_ERRORS.labels(method=extraction_method).inc()
        INFERENCE_REQUESTS.labels(method=extraction_method, status="error").inc()
        INFERENCE_LATENCY.observe(time.perf_counter() - start_time)
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        INFERENCE_ERRORS.labels(method=extraction_method).inc()
        INFERENCE_REQUESTS.labels(method=extraction_method, status="error").inc()
        INFERENCE_LATENCY.observe(time.perf_counter() - start_time)
        logger.exception("Error during prediction")
        detail = (
            f"Internal server error: {e!s}"
            if os.getenv("EXPOSE_INTERNAL_ERRORS", "false").lower()
            in {"1", "true", "yes", "on"}
            else "Internal server error"
        )
        raise HTTPException(status_code=500, detail=detail)
