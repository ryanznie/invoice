import base64

import pytest

from scripts.smoke_test_runpod_backend import build_payload, summarize


def test_build_payload_encodes_files(tmp_path):
    image_path = tmp_path / "invoice.jpg"
    ocr_path = tmp_path / "invoice.txt"
    image_path.write_bytes(b"image-bytes")
    ocr_path.write_bytes(b"0,0,1,0,1,1,0,1,INV-42")

    payload = build_payload(image_path, ocr_path)

    assert payload["image_filename"] == "invoice.jpg"
    assert payload["ocr_filename"] == "invoice.txt"
    assert base64.b64decode(payload["image_base64"]) == b"image-bytes"
    assert base64.b64decode(payload["ocr_base64"]) == b"0,0,1,0,1,1,0,1,INV-42"


def test_build_payload_rejects_ocr_without_coordinates_format(tmp_path):
    image_path = tmp_path / "invoice.jpg"
    ocr_path = tmp_path / "invoice.csv"
    image_path.write_bytes(b"image-bytes")
    ocr_path.write_bytes(b"ocr")

    with pytest.raises(ValueError, match=r"\.txt or \.json"):
        build_payload(image_path, ocr_path)


def test_summarize_includes_only_invoice_number_words():
    result = summarize(
        {
            "invoice_number": "INV-42",
            "extraction_method": "heuristic",
            "model_device": "cpu",
            "total_words": 2,
            "predictions": [
                {"word": "Invoice", "is_invoice_number": False},
                {"word": "INV-42", "is_invoice_number": True},
            ],
        }
    )

    assert result == {
        "invoice_number": "INV-42",
        "extraction_method": "heuristic",
        "model_device": "cpu",
        "total_words": 2,
        "matched_words": ["INV-42"],
    }
