#!/usr/bin/env bash

set -euo pipefail

APP_PORT="${PORT:-7860}"
APP_HOST="${HOST:-0.0.0.0}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8000}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8001}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-8002}"
TRITON_MODEL_REPOSITORY="${TRITON_MODEL_REPOSITORY:-/app/triton_model_repo}"

echo "Starting Triton on ports ${TRITON_HTTP_PORT}/${TRITON_GRPC_PORT}/${TRITON_METRICS_PORT}"
tritonserver \
  --model-repository="${TRITON_MODEL_REPOSITORY}" \
  --http-port="${TRITON_HTTP_PORT}" \
  --grpc-port="${TRITON_GRPC_PORT}" \
  --metrics-port="${TRITON_METRICS_PORT}" &

TRITON_PID=$!

cleanup() {
  echo "Stopping Triton..."
  kill "${TRITON_PID}" || true
}

trap cleanup EXIT

echo "Waiting for Triton to initialize..."
until curl -sf "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready" >/dev/null; do
  sleep 2
done

echo "Starting FastAPI on ${APP_HOST}:${APP_PORT}"
python app.py --host "${APP_HOST}" --port "${APP_PORT}"
