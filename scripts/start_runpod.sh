#!/usr/bin/env bash

set -euo pipefail

APP_PORT="${PORT:-7860}"
APP_HOST="${HOST:-0.0.0.0}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8000}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8001}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-8002}"
TRITON_MODEL_NAME="${TRITON_MODEL_NAME:-layoutlmv3-lora-invoice-number}"
TRITON_MODEL_REPOSITORY="${TRITON_MODEL_REPOSITORY:-/workspace/triton_model_repo}"
MODEL_ONNX_PATH="${MODEL_ONNX_PATH:-/workspace/models/artifacts/layoutlmv3_invoice_ner.onnx}"
MODEL_VERSION="${TRITON_MODEL_VERSION:-1}"

if [[ ! -f "${MODEL_ONNX_PATH}" ]]; then
  echo "Model artifact is missing: ${MODEL_ONNX_PATH}" >&2
  echo "Place the ONNX file on the mounted Runpod volume before starting this Pod." >&2
  exit 1
fi

MODEL_DIR="${TRITON_MODEL_REPOSITORY}/${TRITON_MODEL_NAME}"
mkdir -p "${MODEL_DIR}/${MODEL_VERSION}"
cp "/app/triton_model_repo/${TRITON_MODEL_NAME}/config.pbtxt" "${MODEL_DIR}/config.pbtxt"
ln -sfn "${MODEL_ONNX_PATH}" "${MODEL_DIR}/${MODEL_VERSION}/model.onnx"

echo "Starting Triton with model repository ${TRITON_MODEL_REPOSITORY}"
tritonserver \
  --model-repository="${TRITON_MODEL_REPOSITORY}" \
  --http-port="${TRITON_HTTP_PORT}" \
  --grpc-port="${TRITON_GRPC_PORT}" \
  --metrics-port="${TRITON_METRICS_PORT}" &

TRITON_PID=$!

cleanup() {
  kill "${TRITON_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for Triton to initialize..."
for _ in $(seq 1 90); do
  if curl --silent --fail "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready" >/dev/null; then
    echo "Triton is ready. Starting FastAPI on ${APP_HOST}:${APP_PORT}"
    exec python3 app.py --host "${APP_HOST}" --port "${APP_PORT}"
  fi
  sleep 2
done

echo "Triton did not become ready within 180 seconds." >&2
exit 1
