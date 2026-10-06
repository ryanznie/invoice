import base64

import pytest

from src import runpod_handler


def test_handler_decodes_files_and_calls_prediction(monkeypatch):
    captured = {}

    def fake_predict(*, image, ocr_file):
        captured["image_name"] = image.filename
        captured["image"] = image.file.read()
        captured["ocr_name"] = ocr_file.filename if ocr_file else None
        captured["ocr"] = ocr_file.file.read() if ocr_file else None
        return {"invoice_number": "INV-42"}

    monkeypatch.setattr(runpod_handler, "predict", fake_predict)
    result = runpod_handler.handler(
        {
            "input": {
                "image_base64": base64.b64encode(b"image-bytes").decode(),
                "image_filename": "invoice.png",
                "ocr_base64": base64.b64encode(b"0,0,1,0,1,1,0,1,INV-42").decode(),
                "ocr_filename": "invoice.txt",
            }
        }
    )

    assert result == {"invoice_number": "INV-42"}
    assert captured == {
        "image_name": "invoice.png",
        "image": b"image-bytes",
        "ocr_name": "invoice.txt",
        "ocr": b"0,0,1,0,1,1,0,1,INV-42",
    }


def test_handler_rejects_invalid_base64():
    with pytest.raises(ValueError, match="not valid base64"):
        runpod_handler.handler(
            {
                "input": {
                    "image_base64": "not-base64",
                    "ocr_base64": base64.b64encode(b"ocr").decode(),
                    "ocr_filename": "invoice.txt",
                }
            }
        )


def test_handler_ignores_unknown_ocr_extension(monkeypatch):
    captured = {}

    def fake_predict(*, image, ocr_file):
        captured["image"] = image.file.read()
        captured["ocr_file"] = ocr_file
        return {"invoice_number": "INV-42"}

    monkeypatch.setattr(runpod_handler, "predict", fake_predict)
    result = runpod_handler.handler(
        {
            "input": {
                "image_base64": base64.b64encode(b"image").decode(),
                "ocr_base64": base64.b64encode(b"ocr").decode(),
                "ocr_filename": "invoice.csv",
            }
        }
    )

    assert result == {"invoice_number": "INV-42"}
    assert captured == {"image": b"image", "ocr_file": None}
