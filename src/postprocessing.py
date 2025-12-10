"""
Postprocessing rules for cleaning up extracted invoice numbers.
Based on patterns from notebooks/04_postprocess.ipynb
"""

import logging

logger = logging.getLogger(__name__)


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
