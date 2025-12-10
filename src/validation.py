"""
Validation logic for invoice NER inputs and outputs.
"""

import re
from typing import List
from PIL import Image


def validate_image(image: Image.Image) -> None:
    """
    Validate image input.

    Args:
        image: PIL Image

    Raises:
        ValueError: If image dimensions are invalid
    """
    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")


def validate_words(words: List[str]) -> None:
    """
    Validate words input.

    Args:
        words: List of OCR words

    Raises:
        ValueError: If words list is empty
        TypeError: If words are not all strings
    """
    if not words:
        raise ValueError("Words list cannot be empty")

    if not all(isinstance(w, str) for w in words):
        raise TypeError("All words must be strings")


def validate_boxes(boxes: List[List[int]], words: List[str]) -> None:
    """
    Validate bounding boxes.

    Args:
        boxes: List of bounding boxes [x0, y0, x1, y1] normalized to 0-1000
        words: List of OCR words (for length checking)

    Raises:
        ValueError: If dimensions mismatch or box format is invalid
        TypeError: If box coordinates are not numeric
    """
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


def validate_model_extraction(invoice_number: str) -> bool:
    """
    Validate model-extracted invoice number.
    Returns True if valid, False if should be rejected.

    Args:
        invoice_number: Extracted invoice number

    Returns:
        True if valid, False if should be rejected
    """
    if not invoice_number:
        return True

    # Reject if contains semicolon
    if ";" in invoice_number:
        return False

    # Must contain alphanumeric characters
    if not re.search(r"[a-zA-Z0-9]", invoice_number):
        return False

    return True
