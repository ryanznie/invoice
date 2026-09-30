# Developer Setup

## Install

Requires Python 3.10+, Git, and uv.

    git clone https://github.com/ryanznie/invoice.git
    cd invoice
    uv sync --extra dev
    cp .env.example .env

The ONNX weight is not stored in Git. A fresh checkout needs access to a configured DVC remote or another copy of the pinned model artifact. The default model path is models/artifacts/layoutlmv3_invoice_ner.onnx; keep its processor files beside it.

Start the API and Gradio UI:

    uv run python -m src.demo --host 127.0.0.1

The API docs are at http://127.0.0.1:7860/docs. See [API usage](API_USAGE.md) for upload formats.

## Data labeling

Download SROIE2019 and place its train/test image and OCR files under data/SROIE2019. Then run the Streamlit app from the data directory:

    cd data
    source ../.venv/bin/activate
    streamlit run app.py

The app reads images from SROIE2019/<split>/img and OCR text from SROIE2019/<split>/box. It writes labels.json, test_labels.json, and ambiguous_edits.log in the current directory.

## Development checks

    uv run pytest
    uv run prek run --all-files

See [Testing](TESTING.md) for E2E and load-test commands. Use the scripts' help output for preprocessing and training options:

    uv run python scripts/preprocess.py --help
    uv run python scripts/train.py --help

## Deployment

The local Docker Compose stack and production Runpod worker have separate requirements. See [Monitoring](../monitoring/README.md) and the [Runpod deployment guide](RUNPOD_CPU_SERVERLESS.md).
