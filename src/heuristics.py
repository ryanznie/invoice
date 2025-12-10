"""
Heuristic pattern matching for invoice number extraction.
Based on patterns from notebooks/01_heuristics.ipynb
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def extract_invoice_heuristics(words: List[str], ocr_lines: List[str] = None) -> Tuple:
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

            # 4. Should not be a date pattern (MM/DD/YYYY or DD/MM/YYYY)
            # Check if it looks like a date with 2 slashes
            if extracted.count("/") == 2:
                # Pattern: digits/digits/digits (likely a date)
                date_pattern = r"^\d{1,2}/\d{1,2}/\d{2,4}$"
                if re.match(date_pattern, extracted):
                    logger.warning(
                        f"✗ Rejected '{extracted}': looks like a date (MM/DD/YYYY or DD/MM/YYYY)"
                    )
                    continue

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
