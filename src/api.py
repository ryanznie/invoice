"""
FastAPI endpoints for Invoice NER API.
"""

import io
import os
import json
import tempfile
import logging
from typing import List
from contextlib import asynccontextmanager
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from . import inference
from .heuristics import extract_invoice_heuristics
from .postprocessing import postprocess_invoice_number
from .validation import validate_model_extraction
from .utils import parse_ocr_text_file, normalize_boxes

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
    description="LayoutLMv3 model for extracting invoice numbers",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """Request model for predictions"""

    words: List[str]
    boxes: List[List[int]]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if inference.model is not None else "unhealthy",
        "model_loaded": inference.model is not None,
        "device": inference.DEVICE,
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Invoice image file (JPG, PNG, etc.)"),
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
    if inference.model is None or inference.processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and validate image
        image_bytes = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image = pil_image.convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        img_width, img_height = pil_image.size

        # Read and parse OCR file
        ocr_bytes = await ocr_file.read()
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
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error parsing OCR file: {str(e)}"
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
            confidence_scores = [1.0] * len(words)
        else:
            # Step 2: Fall back to model
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
