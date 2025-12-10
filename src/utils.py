"""
Utility functions for OCR data parsing and coordinate normalization.
"""

import os
import logging
from typing import List, Dict
from scripts import split_invoice_string, estimate_word_boxes

logger = logging.getLogger(__name__)


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
