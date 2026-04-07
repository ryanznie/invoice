"""
FastAPI endpoints for Invoice NER API.
"""

import io
import json
import logging
import time
from typing import List, Optional
from contextlib import asynccontextmanager
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from prometheus_client import make_asgi_app

from . import inference
from . import monitoring
from .heuristics import extract_invoice_heuristics
from .postprocessing import postprocess_invoice_number
from .validation import validate_model_extraction
from .utils import parse_ocr_text_content, normalize_boxes

logger = logging.getLogger(__name__)


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

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


class PredictionRequest(BaseModel):
    """Request model for predictions"""

    words: List[str]
    boxes: List[List[int]]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if inference.backend is not None else "unhealthy",
        "model_loaded": inference.backend is not None,
        "device": inference.DEVICE,
    }


@app.post("/predict")
def predict(
    image: Optional[UploadFile] = File(
        None,
        description="Invoice image file (JPG, PNG, etc.). Optional for raw OCR text without coordinates.",
    ),
    ocr_file: UploadFile = File(..., description="OCR data file (TXT or JSON format)"),
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
    try:
        pil_image = None
        img_width = None
        img_height = None
        if image is not None:
            image_bytes = image.file.read()
            try:
                pil_image = Image.open(io.BytesIO(image_bytes))
                pil_image = pil_image.convert("RGB")
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid image file: {str(e)}"
                )

            img_width, img_height = pil_image.size

        # Read and parse OCR file
        ocr_bytes = ocr_file.file.read()
        ocr_filename = ocr_file.filename.lower()
        raw_text = None
        has_boxes = False

        try:
            if ocr_filename.endswith(".json"):
                # Parse JSON file
                ocr_data = json.loads(ocr_bytes.decode("utf-8"))
                words = ocr_data.get("words", [])
                boxes = ocr_data.get("bboxes", ocr_data.get("boxes", []))
                ocr_lines = ocr_data.get("ocr_lines", None)
                raw_text = ocr_data.get("raw_text")
                has_boxes = bool(boxes)

                # Check if boxes need normalization
                needs_normalization = (
                    any(coord > 1000 for box in boxes for coord in box)
                    if boxes
                    else False
                )
                if needs_normalization:
                    if pil_image is None or img_width is None or img_height is None:
                        raise HTTPException(
                            status_code=400,
                            detail="Image file is required to normalize pixel-coordinate OCR data",
                        )
                    logger.info(
                        "Detected pixel coordinates in JSON, normalizing to 0-1000 range"
                    )
                    boxes = normalize_boxes(boxes, img_width, img_height)

            elif ocr_filename.endswith(".txt"):
                ocr_data = parse_ocr_text_content(
                    ocr_bytes.decode("utf-8", errors="ignore")
                )
                words = ocr_data["words"]
                boxes = ocr_data["bboxes"]
                ocr_lines = ocr_data.get("ocr_lines", None)
                raw_text = ocr_data.get("raw_text")
                has_boxes = ocr_data.get("has_boxes", False)

                if has_boxes:
                    if pil_image is None or img_width is None or img_height is None:
                        raise HTTPException(
                            status_code=400,
                            detail="Image file is required when OCR text includes coordinates",
                        )
                    boxes = normalize_boxes(boxes, img_width, img_height)
            else:
                raise HTTPException(
                    status_code=400, detail="OCR file must be .txt or .json format"
                )

        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error parsing OCR file: {str(e)}"
            )

        if not words:
            raise HTTPException(status_code=400, detail="OCR file must contain text")

        logger.info("=" * 60)
        logger.info("API: Starting invoice extraction pipeline")
        logger.info(f"Total words: {len(words)}, Total boxes: {len(boxes)}")
        if ocr_lines:
            logger.info(f"OCR lines available: {len(ocr_lines)}")
        logger.info("=" * 60)

        # Step 1: Try heuristics first
        start_time = time.time()
        invoice_number, matched_indices = extract_invoice_heuristics(words, ocr_lines)

        if invoice_number:
            # Heuristic found a match
            extraction_method = "heuristic"
            logger.info(f"Using heuristic extraction result: '{invoice_number}'")
            labels = [
                "HEURISTIC_MATCH" if i in matched_indices else "LABEL_0"
                for i in range(len(words))
            ]
            confidence_scores = [1.0] * len(words)

            monitoring.record_inference_metrics(
                method="heuristic",
                status="success",
                duration=time.time() - start_time,
            )

        elif has_boxes:
            # Step 2: Fall back to model
            if pil_image is None:
                raise HTTPException(
                    status_code=400,
                    detail="Image file is required when heuristics fail on coordinate-based OCR input",
                )
            extraction_method = "model"
            logger.info("🤖 Falling back to LayoutLMv3 model inference")
            result = inference.predict_invoice(pil_image, words, boxes)
            invoice_number = result["invoice_number"]
            labels = result["labels"]
            confidence_scores = result["confidence_scores"]
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
        else:
            extraction_method = "gemini"
            logger.info(
                "✍️ No coordinates provided. Falling back to Gemini text extraction"
            )
            monitoring.record_fallback("no_coordinates", "gemini")

            if inference.gemini_client is None:
                raise HTTPException(
                    status_code=503, detail="Gemini fallback is not initialized"
                )

            result = inference.gemini_client.predict(
                image=pil_image,
                words=words,
                raw_text=raw_text,
            )

            invoice_number = result.get("invoice_number")
            labels = ["LABEL_0"] * len(words)
            confidence_scores = [0.0] * len(words)

            if result.get("error"):
                monitoring.record_error("GeminiTextFallbackError")
                monitoring.record_inference_metrics(
                    method="gemini",
                    status="error",
                    duration=time.time() - start_time,
                )
                raise HTTPException(
                    status_code=502,
                    detail=f"Gemini fallback failed: {result['error']}",
                )

            if invoice_number:
                invoice_number = postprocess_invoice_number(invoice_number)
                if not validate_model_extraction(invoice_number):
                    invoice_number = None

            monitoring.record_inference_metrics(
                method="gemini",
                status="success" if invoice_number else "error",
                duration=time.time() - start_time,
            )

        # Final result
        invoice_number = invoice_number or "Not Found"
        logger.info("=" * 60)
        logger.info(f"API RESULT: '{invoice_number}' (via {extraction_method})")
        logger.info("=" * 60)

        # Build detailed predictions
        predictions = []
        for i, (word, label, conf) in enumerate(
            zip(words[: len(labels)], labels, confidence_scores)
        ):
            predictions.append(
                {
                    "word": word,
                    "label": label,
                    "confidence": round(conf, 4),
                    "is_invoice_number": label.startswith("LABEL_1")
                    or label.startswith("LABEL_2")
                    or label == "HEURISTIC_MATCH",
                }
            )

        return {
            "invoice_number": invoice_number,
            "extraction_method": extraction_method,
            "predictions": predictions,
            "total_words": len(words),
            "model_device": inference.DEVICE,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
