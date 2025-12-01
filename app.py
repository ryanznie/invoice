"""
FastAPI + Gradio app for Invoice NER model testing
Accepts: Image + Text file (JSON with words and bboxes)
"""

import os
import json
import torch
import logging
from PIL import Image
from typing import List, Dict
from contextlib import asynccontextmanager
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from peft import PeftModel
from scripts import split_invoice_string, estimate_word_boxes
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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

    # Validate image
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image).__name__}")

    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")

    # Validate words
    if not isinstance(words, list):
        raise TypeError(f"Expected list for words, got {type(words).__name__}")

    if not words:
        raise ValueError("Words list cannot be empty")

    if not all(isinstance(w, str) for w in words):
        raise TypeError("All words must be strings")

    # Validate boxes
    if not isinstance(boxes, list):
        raise TypeError(f"Expected list for boxes, got {type(boxes).__name__}")

    if len(words) != len(boxes):
        raise ValueError(f"Mismatch: {len(words)} words but {len(boxes)} boxes")

    # Validate box format
    for i, box in enumerate(boxes):
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Box {i} must be a list of 4 integers, got {box}")

        if not all(isinstance(coord, (int, float)) for coord in box):
            raise TypeError(f"Box {i} coordinates must be numeric, got {box}")

        # Validate normalized coordinates (0-1000 range)
        if not all(0 <= coord <= 1000 for coord in box):
            raise ValueError(
                f"Box {i} coordinates must be in range [0, 1000], got {box}"
            )

        # Validate box geometry (x0 < x1, y0 < y1)
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(
                f"Box {i} has invalid geometry (x0={box[0]}, y0={box[1]}, x1={box[2]}, y1={box[3]}). Expected x0<x1 and y0<y1"
            )

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
    # print(invoice_tokens)
    return {
        "words": words[: len(predicted_labels)],
        "labels": predicted_labels,
        "invoice_number": invoice_number,
        "confidence_scores": word_confidences,
    }


# ============================================================================
# FASTAPI APP
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    load_model()
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
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": DEVICE,
    }


# ============================================================================
# GRADIO INTERFACE
# ============================================================================


def parse_ocr_text_file(file_path: str) -> Dict:
    """
    Parse OCR text file format to JSON
    Uses functions from scripts/preprocess.py for consistency with training data.

    Format: x1,y1,x2,y2,x3,y3,x4,y4,text
    Example: 83,41,331,41,331,78,83,78,TAN WOON YANN

    Steps:
    1. Parse each line as a text box
    2. Split multi-word lines into tokens (split_invoice_string)
    3. Estimate individual word bounding boxes (estimate_word_boxes)

    Args:
        file_path: Path to text file

    Returns:
        Dictionary with words and bboxes

    Raises:
        ValueError: If file is empty or has invalid format
        FileNotFoundError: If file does not exist
    """
    # Validate file path
    if not isinstance(file_path, str):
        raise TypeError(
            f"Expected string for file_path, got {type(file_path).__name__}"
        )

    if not file_path:
        raise ValueError("File path cannot be empty")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    ocr_entries = []

    # Parse file into entries
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue

            coords = list(map(int, parts[:8]))
            text = ",".join(parts[8:]).strip()

            if not text:
                continue

            # Convert 4-point polygon to bounding box [x0, y0, x1, y1]
            xs, ys = coords[::2], coords[1::2]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            ocr_entries.append({"text": text, "bbox": bbox})

    # Sort by y then x coordinate to ensure reading order
    ocr_entries.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))

    # Keep original lines for heuristic search
    ocr_lines = [entry["text"] for entry in ocr_entries]

    # Split multi-word lines and estimate word boxes
    words = []
    boxes = []

    for entry in ocr_entries:
        text = entry["text"]
        line_bbox = entry["bbox"]

        # Split text into tokens (handles delimiters properly)
        tokens = split_invoice_string(text)

        # Estimate individual word boxes from line box
        token_boxes = estimate_word_boxes(text, tokens, line_bbox)

        # Add words with their estimated boxes
        for token, token_bbox in zip(tokens, token_boxes):
            words.append(token)
            boxes.append(token_bbox)

    return {"words": words, "bboxes": boxes, "ocr_lines": ocr_lines}


def normalize_boxes(
    boxes: List[List[int]], image_width: int, image_height: int
) -> List[List[int]]:
    """
    Normalize bounding boxes to 0-1000 range

    Args:
        boxes: List of boxes in pixel coordinates [x0, y0, x1, y1]
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        List of normalized boxes (clamped to [0, 1000])

    Raises:
        ValueError: If image dimensions are invalid or boxes are malformed
        TypeError: If inputs have incorrect types
    """
    # Validate inputs
    if not isinstance(boxes, list):
        raise TypeError(f"Expected list for boxes, got {type(boxes).__name__}")

    if not isinstance(image_width, (int, float)) or not isinstance(
        image_height, (int, float)
    ):
        raise TypeError("Image dimensions must be numeric")

    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Invalid image dimensions: width={image_width}, height={image_height}"
        )

    if not boxes:
        return []
    normalized = []
    for i, box in enumerate(boxes):
        # Calculate raw normalized values
        raw_normalized = [
            int(box[0] * 1000 / image_width),
            int(box[1] * 1000 / image_height),
            int(box[2] * 1000 / image_width),
            int(box[3] * 1000 / image_height),
        ]

        # Clamp to [0, 1000] range
        normalized_box = [
            max(0, min(1000, raw_normalized[0])),
            max(0, min(1000, raw_normalized[1])),
            max(0, min(1000, raw_normalized[2])),
            max(0, min(1000, raw_normalized[3])),
        ]

        # Ensure valid geometry after clamping (x0 < x1, y0 < y1)
        # If coordinates are equal after clamping, adjust to maintain minimum 1-pixel difference
        if normalized_box[0] >= normalized_box[2]:
            if normalized_box[2] < 1000:
                normalized_box[2] = normalized_box[0] + 1
            else:
                normalized_box[0] = max(0, normalized_box[2] - 1)

        if normalized_box[1] >= normalized_box[3]:
            if normalized_box[3] < 1000:
                normalized_box[3] = normalized_box[1] + 1
            else:
                normalized_box[1] = max(0, normalized_box[3] - 1)

        # Log warning if clamping or adjustment occurred
        if raw_normalized != normalized_box:
            logger.warning(
                f"Box {i} adjusted: pixel {box} -> raw normalized {raw_normalized} -> final {normalized_box} "
                f"(image size: {image_width}x{image_height})"
            )

        normalized.append(normalized_box)
    return normalized


def extract_invoice_heuristics(words: List[str], ocr_lines: List[str] = None) -> tuple:
    """
    Apply heuristic patterns to extract invoice numbers from OCR text.
    Based on patterns from notebooks/01_heuristics.ipynb

    Args:
        words: List of OCR words
        ocr_lines: Optional list of original OCR text lines (before word splitting)

    Returns:
        Tuple of (invoice_number, matched_word_indices) or (None, [])

    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If words list is empty
    """
    # Validate inputs
    if not isinstance(words, list):
        raise TypeError(f"Expected list for words, got {type(words).__name__}")

    if not words:
        raise ValueError("Words list cannot be empty")

    if not all(isinstance(w, str) for w in words):
        raise TypeError("All words must be strings")

    if ocr_lines is not None:
        if not isinstance(ocr_lines, list):
            raise TypeError(
                f"Expected list for ocr_lines, got {type(ocr_lines).__name__}"
            )

        if not all(isinstance(line, str) for line in ocr_lines):
            raise TypeError("All OCR lines must be strings")
    import re

    logger.info(f"🎯 Starting heuristic extraction on {len(words)} words")

    # If we have original OCR lines, search those first (more accurate)
    # Otherwise fall back to joining words
    if ocr_lines:
        logger.debug(f"Searching {len(ocr_lines)} OCR lines")
        text = "\n".join(ocr_lines)
    else:
        logger.debug("No OCR lines provided, joining words")
        text = " ".join(words)

    logger.debug(f"Text to search (first 1000 chars): {text[:1000]}...")

    # Define heuristic patterns (ordered by specificity)
    # TODO: Put all the heuristics in a separate file and import it into app.py
    patterns = [
        (r"INV#\s*:?\s*([A-Za-z0-9/\-]+)", "INV#"),
        (r"INV:\s*([^\s]+)", "INV:"),
        (r"INV-NO\.\s*([^\s]+)", "INV-NO"),
        (r"INV\s*NO\.?\s*[:\-]?\s*([A-Za-z0-9/\-]+)", "INV NO"),
        (r"INVOICE\s*NO\s*[:\-]?\s*([A-Za-z0-9/\-_]+)", "INVOICE NO"),
        (r"INVOICE\s*#\s*:?[\s]*([A-Za-z0-9/\-]+)", "INVOICE#"),
        (r"SLIP\s*(?:NO\.?|NUMBER)?\s*[^A-Za-z0-9]*\s*([A-Za-z0-9/\-]+)", "SLIP"),
        (r"RECEIPT\s*NO\s*[:\-]?\s*([A-Za-z0-9/\-]+)", "RECEIPT NO"),
        (r"BILL\s*NO\s*[:\-]?\s*([A-Za-z0-9/\-]+)", "BILL NO"),
        (r"CB#\s*:\s*([A-Za-z0-9/\-]+)", "CB#"),
        (r"C/N\s*NO\s*[:\-]?\s*([A-Za-z0-9/\-]+)", "C/N NO"),
        (r"TRANSACTION\s*NO\s*[:\-]?\s*([A-Za-z0-9/\-]+)", "TRANSACTION NO"),
        (r"TRN\s*[:\-]?\s*([A-Za-z0-9/\-]+)", "TRN"),
        (r"RCPT#\s*:\s*([A-Za-z0-9]+)", "RCPT"),
    ]

    # Try each pattern
    for pattern, name in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted = match.group(1)
            logger.info(f"✓ Pattern '{name}' matched: '{extracted}'")

            # Validation rules from notebook:
            # 1. Must be > 3 characters
            if len(extracted) <= 3:
                logger.warning(f"✗ Rejected '{extracted}': too short (<= 3 chars)")
                continue

            # 2. Must contain at least one digit
            if not re.search(r"\d", extracted):
                logger.warning(f"✗ Rejected '{extracted}': no digits found")
                continue

            # 3. Should not contain multiple comma-separated values
            if "," in extracted:
                original = extracted
                extracted = extracted.split(",")[0]
                logger.info(
                    f"⚠ Multiple values detected, taking first: '{original}' → '{extracted}'"
                )

            # Find the position of the extracted invoice number in the text
            # Since heuristics are accurate, we just need to find where it appears
            matched_indices = []
            match_start = match.start(1)  # Start position of captured group
            match_end = match.end(1)  # End position of captured group

            # Find which words fall within the matched text range
            current_pos = 0
            for i, word in enumerate(words):
                word_start = text.find(word, current_pos)
                word_end = word_start + len(word)

                # Check if this word overlaps with the matched invoice number
                if word_start < match_end and word_end > match_start:
                    matched_indices.append(i)

                current_pos = word_end
                if current_pos >= match_end:
                    break

            logger.info(
                f"✅ Heuristic extraction successful: '{extracted}' (matched word indices: {matched_indices})"
            )
            return extracted, matched_indices

    logger.info("❌ No heuristic patterns matched")
    return None, []


def postprocess_invoice_number(invoice_number: str) -> str:
    """
    Apply postprocessing rules to clean up extracted invoice numbers.
    Based on patterns from notebooks/04_postprocess.ipynb

    Args:
        invoice_number: Raw extracted invoice number

    Returns:
        Cleaned invoice number

    Raises:
        TypeError: If invoice_number is not a string
    """
    # Validate input
    if invoice_number is not None and not isinstance(invoice_number, str):
        raise TypeError(
            f"Expected string or None for invoice_number, got {type(invoice_number).__name__}"
        )
    if not invoice_number:
        return invoice_number

    logger.info(f"✨ Starting postprocessing: '{invoice_number}'")
    original = invoice_number

    # Remove colons and strip
    if ":" in invoice_number:
        invoice_number = invoice_number.replace(":", "").strip()
        logger.info(f"  → Removed colons: '{invoice_number}'")

    # Replace ' - ' with '-'
    if " - " in invoice_number:
        invoice_number = invoice_number.replace(" - ", "-")
        logger.info(f"  → Normalized dashes: '{invoice_number}'")

    # Replace ' / ' with '/'
    if " / " in invoice_number:
        invoice_number = invoice_number.replace(" / ", "/")
        logger.info(f"  → Normalized slashes: '{invoice_number}'")

    # Replace 'SP NULL' with 'SP-NULL'
    if "SP NULL" in invoice_number:
        invoice_number = invoice_number.replace("SP NULL", "SP-NULL")
        logger.info(f"  → Fixed SP NULL: '{invoice_number}'")

    # If contains SP-NULL, only take first 24 characters
    if "SP-NULL" in invoice_number and len(invoice_number) > 24:
        invoice_number = invoice_number[:24]
        logger.info(f"  → Truncated SP-NULL to 24 chars: '{invoice_number}'")

    # Remove 'DATE' suffix if present
    if invoice_number.endswith("DATE"):
        invoice_number = invoice_number[:-4].strip()
        logger.info(f"  → Removed DATE suffix: '{invoice_number}'")

    if original != invoice_number:
        logger.info(f"✅ Postprocessing complete: '{original}' → '{invoice_number}'")
    else:
        logger.info("✅ Postprocessing complete: no changes needed")

    return invoice_number


def gradio_predict(image, text_file):
    """
    Gradio prediction function

    Args:
        image: Uploaded image
        text_file: Uploaded text/JSON file with OCR data

    Returns:
        Formatted results string and highlighted image
    """
    if image is None:
        return "Please upload an image", None, ""

    if text_file is None:
        return "Please upload a text file with OCR data", None, ""

    try:
        # Convert image to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Get image dimensions
        img_width, img_height = image.size

        # Determine file type and parse accordingly
        file_path = text_file.name

        if file_path.endswith(".json"):
            # Read JSON file
            with open(file_path, "r") as f:
                ocr_data = json.load(f)

            # Extract words and boxes
            words = ocr_data.get("words", [])
            boxes = ocr_data.get("bboxes", ocr_data.get("boxes", []))
            ocr_lines = ocr_data.get("ocr_lines", None)  # May not be present in JSON

            # Check if boxes need normalization (if any coordinate > 1000, assume pixel coordinates)
            needs_normalization = (
                any(coord > 1000 for box in boxes for coord in box) if boxes else False
            )

            if needs_normalization:
                logger.info(
                    "Detected pixel coordinates in JSON, normalizing to 0-1000 range"
                )
                boxes = normalize_boxes(boxes, img_width, img_height)
        else:
            # Parse text file format
            ocr_data = parse_ocr_text_file(file_path)
            words = ocr_data["words"]
            boxes = ocr_data["bboxes"]
            ocr_lines = ocr_data.get("ocr_lines", None)

            # Normalize boxes to 0-1000 range
            boxes = normalize_boxes(boxes, img_width, img_height)

        if not words or not boxes:
            return (
                "File must contain valid OCR data",
                None,
                "",
            )

        logger.info("=" * 60)
        logger.info("Starting invoice extraction pipeline")
        logger.info(f"Total words: {len(words)}, Total boxes: {len(boxes)}")
        if ocr_lines:
            logger.info(f"OCR lines available: {len(ocr_lines)}")
        logger.info("=" * 60)

        # Step 1: Try heuristics first (with original OCR lines if available)
        invoice_number, matched_indices = extract_invoice_heuristics(words, ocr_lines)

        if invoice_number:
            # Heuristic found a match
            extraction_method = "🎯 Heuristic"
            logger.info(f"Using heuristic extraction result: '{invoice_number}'")
            # Create labels for visualization (green boxes for matched words)
            labels = [
                "HEURISTIC_MATCH" if i in matched_indices else "LABEL_0"
                for i in range(len(words))
            ]
            result = {
                "words": words,
                "labels": labels,
                "confidence_scores": [1.0] * len(words),
                "matched_indices": matched_indices,
            }
        else:
            # Step 2: Fall back to model
            extraction_method = "🤖 Model"
            logger.info("🤖 Falling back to LayoutLMv3 model inference")
            result = predict_invoice(image, words, boxes)
            invoice_number = result["invoice_number"]
            logger.info(f"Model extracted: '{invoice_number}'")

            # Step 3: Apply postprocessing to model results
            if invoice_number:
                invoice_number = postprocess_invoice_number(invoice_number)

        # Final display
        invoice_number = invoice_number or "Not Found"
        logger.info("=" * 60)
        logger.info(f"FINAL RESULT: '{invoice_number}' (via {extraction_method})")
        logger.info("=" * 60)

        # Create annotated image and detailed output
        if extraction_method == "🎯 Heuristic":
            # Heuristic extraction - show green boxes for matched words
            annotated_image = create_annotated_image(
                image, words, boxes, result["labels"]
            )
            detailed_output = f"## 📊 Extraction Method: {extraction_method}\n\n"
            detailed_output += (
                "Invoice number extracted using **pattern matching heuristics**.\n\n"
            )
            detailed_output += "✨ Fast extraction without model inference.\n\n"
            detailed_output += f"🔴 **{len(result['matched_indices'])} word(s)** matched the heuristic pattern.\n"

            # Warning if multiple words matched
            if len(result["matched_indices"]) >= 2:
                detailed_output += "\n⚠️ **WARNING**: Multiple words detected. Please inspect carefully to ensure accuracy.\n"
                logger.warning(
                    f"Multiple words ({len(result['matched_indices'])}) matched heuristic pattern"
                )
        else:
            # Model extraction - show annotations and predictions
            annotated_image = create_annotated_image(
                image, words, boxes, result["labels"]
            )

            # Count predicted invoice number words
            predicted_count = sum(
                1 for label in result["labels"] if label.startswith("LABEL_1")
            )

            detailed_output = f"## 📊 Extraction Method: {extraction_method}\n\n"
            detailed_output += "Invoice number extracted using LayoutLMv3 model.\n\n"

            # Warning if multiple predictions
            if predicted_count >= 2:
                detailed_output += "⚠️ **WARNING**: Multiple words predicted as invoice number. Please inspect carefully to ensure accuracy.\n\n"
                logger.warning(
                    f"Multiple words ({predicted_count}) predicted as invoice number by model"
                )

            detailed_output += "### 📋 Word-level Predictions\n\n"
            detailed_output += "| Word | Label | Confidence |\n"
            detailed_output += "|------|-------|------------|\n"

            for word, label, conf in zip(
                result["words"], result["labels"], result["confidence_scores"]
            ):
                emoji = (
                    "🟢"
                    if label.startswith("LABEL_1") or label.startswith("LABEL_2")
                    else "⚪"
                )
                detailed_output += f"| {emoji} {word} | `{label}` | {conf:.3f} |\n"

        return invoice_number, annotated_image, detailed_output

    except Exception as e:
        return f"❌ Error: {str(e)}", None, ""


def create_annotated_image(image, words, boxes, labels):
    """
    Create an annotated image with bounding boxes

    Args:
        image: PIL Image
        words: List of words
        boxes: List of bounding boxes
        labels: List of predicted labels (may be shorter than words/boxes)

    Returns:
        Annotated PIL Image

    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image).__name__}")

    if (
        not isinstance(words, list)
        or not isinstance(boxes, list)
        or not isinstance(labels, list)
    ):
        raise TypeError("words, boxes, and labels must be lists")

    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")
    from PIL import ImageDraw

    # Create a copy
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Get image dimensions
    img_width, img_height = img.size

    # Handle case where labels list is shorter (e.g., heuristic extraction)
    num_to_annotate = min(len(words), len(boxes), len(labels))

    for i in range(num_to_annotate):
        # word = words[i]
        box = boxes[i]
        label = labels[i]

        # Denormalize coordinates (boxes are in 0-1000 range)
        x0 = int(box[0] * img_width / 1000)
        y0 = int(box[1] * img_height / 1000)
        x1 = int(box[2] * img_width / 1000)
        y1 = int(box[3] * img_height / 1000)

        # Choose color based on label
        if (
            label == "HEURISTIC_MATCH"
            or label.startswith("LABEL_1")
            or label.startswith("LABEL_2")
        ):
            # Red for both heuristic matches and model predictions
            color = "red"
            width_box = 3
        else:
            # Light blue for other words
            color = "lightblue"
            width_box = 1

        # Draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width_box)

    return img


# Create Gradio interface
with gr.Blocks(title="Invoice NER Demo") as demo:
    gr.Markdown("# 🧾 Invoice Number Extraction")

    # How to Use section at the top
    gr.Markdown("## 💡 How to Use")
    gr.Markdown(
        """
        1. Upload an invoice image (JPG, PNG)
        2. Upload OCR data:
           - **Text file (.txt)**: One line per word in format `x1,y1,x2,y2,x3,y3,x4,y4,text`
           - **JSON file (.json)**: With `words` and `bboxes` fields
        3. Click "Extract Invoice Number"
        
        **Hybrid Extraction:**
        - 🎯 **Heuristics First**: Tries pattern matching (e.g., "INV#", "INVOICE NO") for fast extraction
        - 🤖 **Model Fallback**: Uses LayoutLMv3 if no heuristic matches
        - ✨ **Postprocessing**: Cleans up results (removes extra spaces, colons, etc.)
        
        **Color Coding:**
        - 🔴 **Red**: Detected invoice numbers (heuristic or model)
        - 🔵 **Light Blue**: Other text
        """
    )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="📷 Upload Invoice Image", type="pil")
            text_input = gr.File(
                label="📄 Upload OCR Data (TXT or JSON)", file_types=[".txt", ".json"]
            )
            predict_btn = gr.Button("🚀 Extract Invoice Number", variant="primary")

        with gr.Column():
            # Large prominent display for invoice number
            invoice_output = gr.Textbox(
                label="🎯 Extracted Invoice Number",
                placeholder="Invoice number will appear here...",
                interactive=False,
                scale=2,
                container=True,
                show_label=True,
            )
            output_image = gr.Image(label="📊 Annotated Image")
            output_text = gr.Markdown(label="📋 Detailed Results")

    # Supported file formats info at bottom
    with gr.Accordion("📝 Supported File Formats", open=False):
        gr.Markdown(
            """
            **Option 1: Text File (.txt)**
            ```
            x1,y1,x2,y2,x3,y3,x4,y4,text
            83,41,331,41,331,78,83,78,TAN WOON YANN
            109,171,330,171,330,191,109,191,MR D.I.Y. (M) SDN BHD
            ```
            
            **Option 2: JSON File (.json)**
            ```json
            {
                "words": ["INVOICE", "NO:", "INV-12345"],
                "bboxes": [[100, 100, 200, 120], [210, 100, 280, 120], ...]
            }
            ```
            """
        )

    # Connect button
    predict_btn.click(
        fn=gradio_predict,
        inputs=[image_input, text_input],
        outputs=[invoice_output, output_image, output_text],
    )


# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="Invoice NER App")
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with auto-reload"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "7860")),
        help="Port to run on (default: from PORT env or 7860)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind to (default: from HOST env or 0.0.0.0)",
    )
    args = parser.parse_args()

    # Load model before starting server (unless in debug mode with reload)
    if not args.debug:
        load_model()

    print("\n" + "=" * 60)
    print("🚀 Starting Invoice NER App")
    if args.debug:
        print("🐛 DEBUG MODE - Auto-reload enabled")
    print("=" * 60)
    print(f"📱 Device: {DEVICE}")
    print(f"🔧 Model: {MODEL_PATH}")
    print(f"🌐 Open your browser to: http://{args.host}:{args.port}")
    print(f"📊 Log Level: {os.getenv('LOG_LEVEL', 'INFO')}")
    print("=" * 60 + "\n")

    uvicorn.run(
        "app:app" if args.debug else app,
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level="debug" if args.debug else "info",
    )
