#!/usr/bin/env python3
"""Exercise the production FastAPI contract and write a repeatable JSON artifact."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = REPO_ROOT / "docs" / "artifacts" / "api-contract-e2e.json"


class _Encoding(dict[str, np.ndarray]):
    """Minimal processor output used only to reach the inference fallback boundary."""

    def word_ids(self, batch_index: int) -> list[int | None]:
        del batch_index
        return [None, 0, 1, None]


class _ProcessorDouble:
    def __call__(self, *args: Any, **kwargs: Any) -> _Encoding:
        del args, kwargs
        return _Encoding(
            pixel_values=np.zeros((1, 3, 2, 2), dtype=np.float32),
            input_ids=np.zeros((1, 4), dtype=np.int64),
            attention_mask=np.ones((1, 4), dtype=np.int64),
            bbox=np.zeros((1, 4, 4), dtype=np.int64),
        )


class _FailingBackend:
    def predict(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        del inputs
        raise RuntimeError("deterministic primary failure")


class _OpenRouterDouble:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls = 0

    def predict(self, *, image: Image.Image, words: list[str]) -> dict[str, Any]:
        del image, words
        self.calls += 1
        return self.result


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(output, format="PNG")
    return output.getvalue()


def _ocr_json(*, heuristic: bool = True) -> bytes:
    words = ["INVOICE", "NO", "INV-12345"] if heuristic else ["REFERENCE", "ABC12345"]
    boxes = [[0, 0, 500, 500] for _ in words]
    return json.dumps({"words": words, "bboxes": boxes}).encode("utf-8")


def _files(image: bytes, ocr: bytes, *, ocr_name: str = "invoice.json") -> dict:
    return {
        "image": ("invoice.png", image, "image/png"),
        "ocr_file": (ocr_name, ocr, "application/octet-stream"),
    }


def _response_body(response: Any) -> Any:
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return response.text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help=f"JSON artifact path (default: {DEFAULT_ARTIFACT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src import inference
    from src.api import MAX_IMAGE_BYTES, MAX_OCR_BYTES, app

    original_backend = inference.backend
    original_processor = inference.processor
    original_openrouter = inference.openrouter_client
    original_expose_errors = os.environ.get("EXPOSE_INTERNAL_ERRORS")
    cases: list[dict[str, Any]] = []

    def check(
        name: str,
        response: Any,
        expected_status: int,
        *,
        expected_detail: str | None = None,
        expected_detail_prefix: str | None = None,
        expected_invoice: str | None = None,
        condition: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        body = _response_body(response)
        passed = response.status_code == expected_status and condition
        if expected_detail is not None:
            passed = passed and body == {"detail": expected_detail}
        if expected_detail_prefix is not None:
            passed = passed and body.get("detail", "").startswith(
                expected_detail_prefix
            )
        if expected_invoice is not None:
            passed = passed and body.get("invoice_number") == expected_invoice
        recorded_body = (
            {"detail_prefix": expected_detail_prefix}
            if expected_detail_prefix is not None
            else body
        )
        cases.append(
            {
                "name": name,
                "passed": passed,
                "expected_status": expected_status,
                "actual_status": response.status_code,
                "response": recorded_body,
                **(metadata or {}),
            }
        )

    client = TestClient(app, raise_server_exceptions=False)
    image = _png_bytes()
    heuristic_ocr = _ocr_json()

    try:
        os.environ["EXPOSE_INTERNAL_ERRORS"] = "false"

        inference.backend = None
        inference.processor = None
        check("readiness_unavailable", client.get("/ping"), 204)
        check(
            "prediction_before_ready",
            client.post("/predict", files=_files(image, heuristic_ocr)),
            503,
            expected_detail="Model not loaded",
        )

        inference.backend = object()
        inference.processor = object()
        check("readiness_available", client.get("/ping"), 200)

        exact_image = image + b"\0" * (MAX_IMAGE_BYTES - len(image))
        check(
            "image_at_limit",
            client.post("/predict", files=_files(exact_image, heuristic_ocr)),
            200,
            expected_invoice="INV-12345",
            metadata={"bytes": len(exact_image)},
        )
        check(
            "image_over_limit",
            client.post(
                "/predict",
                files=_files(exact_image + b"\0", heuristic_ocr),
            ),
            413,
            expected_detail=(
                f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
            ),
            metadata={"bytes": len(exact_image) + 1},
        )

        exact_ocr = heuristic_ocr + b" " * (MAX_OCR_BYTES - len(heuristic_ocr))
        check(
            "ocr_at_limit",
            client.post("/predict", files=_files(image, exact_ocr)),
            200,
            expected_invoice="INV-12345",
            metadata={"bytes": len(exact_ocr)},
        )
        check(
            "ocr_over_limit",
            client.post(
                "/predict",
                files=_files(image, exact_ocr + b" "),
            ),
            413,
            expected_detail=(
                f"OCR file exceeds the {MAX_OCR_BYTES // (1024 * 1024)} MB limit"
            ),
            metadata={"bytes": len(exact_ocr) + 1},
        )

        check(
            "invalid_image",
            client.post(
                "/predict",
                files=_files(b"not-an-image", heuristic_ocr),
            ),
            400,
            expected_detail_prefix="Invalid image file:",
        )
        check(
            "malformed_ocr_json",
            client.post("/predict", files=_files(image, b"{")),
            400,
            expected_detail_prefix="Invalid JSON file:",
        )
        check(
            "empty_ocr_content",
            client.post("/predict", files=_files(image, b"{}")),
            400,
            expected_detail="OCR file must contain valid words and bboxes",
        )
        check(
            "unsupported_ocr_extension",
            client.post(
                "/predict",
                files=_files(image, heuristic_ocr, ocr_name="invoice.csv"),
            ),
            400,
            expected_detail="OCR file must be .txt or .json format",
        )

        model_ocr = _ocr_json(heuristic=False)
        inference.backend = _FailingBackend()
        inference.processor = _ProcessorDouble()
        inference.openrouter_client = None
        check(
            "primary_failure_without_openrouter",
            client.post("/predict", files=_files(image, model_ocr)),
            500,
            expected_detail="Internal server error",
            metadata={"hosted_requests": 0},
        )

        successful_fallback = _OpenRouterDouble(
            {
                "invoice_number": "FALLBACK-42",
                "method": "local-e2e-double",
            }
        )
        inference.openrouter_client = successful_fallback
        response = client.post("/predict", files=_files(image, model_ocr))
        check(
            "openrouter_fallback_success",
            response,
            200,
            expected_invoice="FALLBACK-42",
            condition=successful_fallback.calls == 1,
            metadata={"fallback_calls": successful_fallback.calls},
        )

        failed_fallback = _OpenRouterDouble(
            {"invoice_number": None, "error": "deterministic fallback failure"}
        )
        inference.openrouter_client = failed_fallback
        response = client.post("/predict", files=_files(image, model_ocr))
        check(
            "openrouter_fallback_failure",
            response,
            500,
            expected_detail="Internal server error",
            condition=failed_fallback.calls == 1,
            metadata={"fallback_calls": failed_fallback.calls},
        )
    finally:
        inference.backend = original_backend
        inference.processor = original_processor
        inference.openrouter_client = original_openrouter
        if original_expose_errors is None:
            os.environ.pop("EXPOSE_INTERNAL_ERRORS", None)
        else:
            os.environ["EXPOSE_INTERNAL_ERRORS"] = original_expose_errors

    passed = all(case["passed"] for case in cases)
    artifact = {
        "schema_version": 1,
        "suite": "invoice-api-contract-e2e",
        "result": "passed" if passed else "failed",
        "limits": {
            "max_image_bytes": MAX_IMAGE_BYTES,
            "max_ocr_bytes": MAX_OCR_BYTES,
        },
        "cases": cases,
        "repeat": "uv run python scripts/e2e_api_contract.py",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    print(f"Artifact: {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
