#!/usr/bin/env python3
"""Build a Runpod payload or smoke-test the queue handler locally."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_payload(image_path: Path, ocr_path: Path) -> dict[str, str]:
    """Encode an invoice image and coordinate-bearing OCR file for Runpod."""
    if not image_path.is_file():
        raise ValueError(f"image file does not exist: {image_path}")
    if not ocr_path.is_file():
        raise ValueError(f"OCR file does not exist: {ocr_path}")
    if ocr_path.suffix.lower() not in {".txt", ".json"}:
        raise ValueError("OCR file must end in .txt or .json")

    return {
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        "image_filename": image_path.name,
        "ocr_base64": base64.b64encode(ocr_path.read_bytes()).decode("ascii"),
        "ocr_filename": ocr_path.name,
    }


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    """Keep smoke-test output readable while preserving useful diagnostics."""
    predictions = result.get("predictions") or []
    matched_words = [
        prediction.get("word")
        for prediction in predictions
        if prediction.get("is_invoice_number")
    ]
    return {
        "invoice_number": result.get("invoice_number"),
        "extraction_method": result.get("extraction_method"),
        "model_device": result.get("model_device"),
        "total_words": result.get("total_words"),
        "matched_words": matched_words,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the Runpod handler locally, or print a payload for "
            "`runpodctl serverless run`."
        )
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--ocr", required=True, type=Path)
    parser.add_argument(
        "--expected",
        help="Exit non-zero unless the extracted invoice number matches this value.",
    )
    parser.add_argument(
        "--payload-only",
        action="store_true",
        help="Print the handler payload without loading or running the model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = build_payload(args.image, args.ocr)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.payload_only:
        json.dump(payload, sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")
        return 0

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from runpod_handler import handler
    from src import inference

    inference.load_model()
    result = handler({"input": payload})
    print(json.dumps(summarize(result), indent=2))

    if args.expected is not None and result.get("invoice_number") != args.expected:
        print(
            "error: expected invoice number "
            f"{args.expected!r}, got {result.get('invoice_number')!r}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
