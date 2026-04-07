"""
Utility functions for OCR data parsing and coordinate normalization.
"""

import os
import logging
import re
from typing import List, Dict
from scripts import split_invoice_string, estimate_word_boxes

logger = logging.getLogger(__name__)


def _build_word_level_ocr_data(ocr_entries: List[Dict]) -> Dict:
    """Convert line-level OCR entries into token-level words and boxes."""
    ocr_entries.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))
    ocr_lines = [entry["text"] for entry in ocr_entries]
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

    return {
        "words": words,
        "bboxes": boxes,
        "ocr_lines": ocr_lines,
        "raw_text": "\n".join(ocr_lines),
        "has_boxes": True,
    }


def _parse_coordinate_ocr_lines(lines: List[str]) -> List[Dict]:
    """Parse OCR lines in the x1,y1,...,x4,y4,text format."""
    ocr_entries = []

    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 9:
            continue

        try:
            coords = list(map(int, parts[:8]))
        except ValueError:
            continue

        text = ",".join(parts[8:]).strip()
        if not text:
            continue

        xs, ys = coords[::2], coords[1::2]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        ocr_entries.append({"text": text, "bbox": bbox})

    return ocr_entries


def _parse_plain_ocr_text(text: str) -> Dict:
    """Parse OCR-like raw text that does not include coordinates."""
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        lowered = stripped.lower()
        if lowered in {"system", "user", "assistant"}:
            continue

        stripped = re.sub(r"<[^>]+>", " ", stripped)
        stripped = stripped.replace("**", " ")
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if not stripped:
            continue

        cleaned_lines.append(stripped)

    words = []
    for line in cleaned_lines:
        words.extend(split_invoice_string(line))

    return {
        "words": words,
        "bboxes": [],
        "ocr_lines": cleaned_lines,
        "raw_text": "\n".join(cleaned_lines),
        "has_boxes": False,
    }


def parse_ocr_text_content(text: str) -> Dict:
    """
    Parse OCR text content with or without coordinates.

    Coordinate format:
        x1,y1,x2,y2,x3,y3,x4,y4,text

    Raw text format:
        free-form OCR output with line breaks and no coordinates.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string for text, got {type(text).__name__}")

    if not text.strip():
        raise ValueError("OCR text content cannot be empty")

    lines = text.splitlines()
    ocr_entries = _parse_coordinate_ocr_lines(lines)
    if ocr_entries:
        return _build_word_level_ocr_data(ocr_entries)

    return _parse_plain_ocr_text(text)


def parse_ocr_text_file(file_path: str) -> Dict:
    """
    Parse OCR text file content with or without coordinates.

    Returns token-level words and, when available, estimated bounding boxes.
    """
    if not isinstance(file_path, str):
        raise TypeError(
            f"Expected string for file_path, got {type(file_path).__name__}"
        )

    if not file_path:
        raise ValueError("File path cannot be empty")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return parse_ocr_text_content(f.read())


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
