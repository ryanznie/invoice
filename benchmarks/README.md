# Benchmarks

The benchmark script compares hybrid, LayoutLMv3, ONNX, and OpenRouter extraction and logs per-invoice and aggregate results to Weights & Biases.

Install the development extra. Authenticate with W&B, or use offline mode:

    uv sync --extra dev
    uv run wandb login

Example:

    uv run invoice-ner-benchmark --model hybrid --data-dir data/test --split test --offline

Use --help for the full CLI options. The script also accepts --device and --model-path.

## Input layout

For the test split, the benchmark expects:

- JSON examples at data/test/test.json
- Labels at data/SROIE2019/test/test_labels.json
- Images and OCR text at data/SROIE2019/test/img and data/SROIE2019/test/box

For training, use data/train/train.json and data/SROIE2019/train/labels.json, img, and box. Each JSON example contains file, words, and bboxes. The preprocessing script can create these JSON files; choose its output path to match this layout. The labeling app saves labels.json and test_labels.json in data/, so copy them into the split directories if the benchmark needs those labels.

Ambiguous labels are skipped. See scripts/preprocess.py --help for dataset preparation.
