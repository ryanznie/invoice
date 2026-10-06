"""Runpod Serverless queue worker for CPU invoice inference."""

import base64
import binascii
import io
import os
from typing import Any

from fastapi import HTTPException, UploadFile

from src import inference
from src.api import MAX_IMAGE_BYTES, MAX_OCR_BYTES, predict


def _decode(value: Any, *, field: str, limit: int) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty base64 string")

    max_encoded_size = ((limit + 2) // 3) * 4 + 4
    if len(value) > max_encoded_size:
        raise ValueError(f"{field} exceeds the configured size limit")

    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field} is not valid base64") from exc

    if len(decoded) > limit:
        raise ValueError(f"{field} exceeds the configured size limit")
    return decoded


def _decode_optional_ocr(value: Any) -> bytes | None:
    if value is None:
        return None
    try:
        return _decode(value, field="ocr_base64", limit=MAX_OCR_BYTES)
    except ValueError:
        return None


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input")
    if not isinstance(payload, dict):
        raise TypeError("input must be an object")

    image_bytes = _decode(
        payload.get("image_base64"), field="image_base64", limit=MAX_IMAGE_BYTES
    )
    ocr_bytes = _decode_optional_ocr(payload.get("ocr_base64"))

    image_filename = os.path.basename(
        str(payload.get("image_filename") or "invoice.png")
    )
    ocr_filename = os.path.basename(str(payload.get("ocr_filename") or "invoice.txt"))
    if not ocr_filename.lower().endswith((".txt", ".json")):
        ocr_bytes = None

    image_upload = UploadFile(filename=image_filename, file=io.BytesIO(image_bytes))
    ocr_upload = (
        UploadFile(filename=ocr_filename, file=io.BytesIO(ocr_bytes))
        if ocr_bytes is not None
        else None
    )

    try:
        return predict(image=image_upload, ocr_file=ocr_upload)
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from exc


if __name__ == "__main__":
    import runpod

    inference.load_model()
    runpod.serverless.start({"handler": handler})
