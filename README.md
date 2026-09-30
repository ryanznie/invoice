# Invoice NER

Extract invoice numbers from invoice images using OCR text and bounding boxes. The app exposes a FastAPI API and a Gradio upload UI; extraction uses heuristics first, then a local LayoutLMv3 ONNX model. OpenRouter fallback is optional and disabled by default.

## Run locally

Requires Python 3.10+ and uv. The large ONNX model is managed outside Git; make it available at the configured model path before starting.

    uv sync --extra dev
    cp .env.example .env
    uv run python -m src.demo --host 127.0.0.1

Open http://127.0.0.1:7860/ for the upload UI or http://127.0.0.1:7860/docs for the API.

To upload a sample invoice and OCR JSON file:

    curl -F 'image=@invoice.jpg' -F 'ocr_file=@ocr.json' http://127.0.0.1:7860/predict

## Tests

    uv run pytest

See [Testing](docs/TESTING.md) for contract and container checks.

## Documentation

- [API usage](docs/API_USAGE.md)
- [Developer setup and data labeling](docs/DEV_SETUP.md)
- [Runpod deployment](docs/RUNPOD_CPU_SERVERLESS.md)
- [Monitoring](monitoring/README.md)
- [Benchmarks](benchmarks/README.md)
- [Model card](models/layoutlmv3-lora-invoice-number/README.md)

Core code is in src/, operational scripts in scripts/, and tests in tests/.
