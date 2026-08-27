#!/usr/bin/env bash

set -euo pipefail

APP_PORT="${PORT:-7860}"
APP_HOST="${HOST:-0.0.0.0}"
MODEL_ONNX_PATH="${MODEL_ONNX_PATH:-/workspace/models/artifacts/layoutlmv3_invoice_ner.onnx}"

if [[ ! -f "${MODEL_ONNX_PATH}" ]]; then
  echo "Model artifact is missing: ${MODEL_ONNX_PATH}" >&2
  echo "Place the ONNX file on the mounted Runpod volume before starting this Pod." >&2
  exit 1
fi

echo "Starting FastAPI CPU ONNX backend on ${APP_HOST}:${APP_PORT}"
exec python3 app.py --host "${APP_HOST}" --port "${APP_PORT}"
