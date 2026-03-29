import os
import logging
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "layoutlmv3-lora-invoice-number")
MODEL_VERSION = os.getenv("TRITON_MODEL_VERSION", "1")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

_LABELS = ["method", "status", "model_name", "model_version", "environment"]

INFERENCE_REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total number of inference requests processed",
    _LABELS,
)

INFERENCE_ERRORS_TOTAL = Counter(
    "inference_errors_total",
    "Total number of failed inference requests",
    ["error_type", "model_name", "model_version", "environment"],
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "Time taken for inference (seconds)",
    ["method", "model_name", "model_version", "environment"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf")),
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Confidence score of the prediction (0-1)",
    ["model_name", "model_version", "environment"],
    buckets=(
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.85,
        0.9,
        0.95,
        0.98,
        0.99,
        1.0,
    ),
)

INPUT_TOKEN_COUNT = Histogram(
    "input_token_count",
    "Number of tokens in the input",
    ["model_name", "model_version", "environment"],
    buckets=(0, 64, 128, 256, 384, 512, 768, 1024),
)

FALLBACK_TOTAL = Counter(
    "fallback_total",
    "Total number of fallbacks triggered",
    ["trigger_reason", "target_backend", "environment"],
)


def record_inference_metrics(method: str, status: str, duration: float) -> None:
    INFERENCE_REQUESTS_TOTAL.labels(
        method=method,
        status=status,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        environment=ENVIRONMENT,
    ).inc()

    INFERENCE_LATENCY_SECONDS.labels(
        method=method,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        environment=ENVIRONMENT,
    ).observe(duration)


def record_error(error_type: str) -> None:
    INFERENCE_ERRORS_TOTAL.labels(
        error_type=error_type,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        environment=ENVIRONMENT,
    ).inc()


def record_ml_metrics(confidence: float, token_count: int) -> None:
    PREDICTION_CONFIDENCE.labels(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        environment=ENVIRONMENT,
    ).observe(confidence)

    INPUT_TOKEN_COUNT.labels(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        environment=ENVIRONMENT,
    ).observe(token_count)


def record_fallback(reason: str, target: str) -> None:
    FALLBACK_TOTAL.labels(
        trigger_reason=reason,
        target_backend=target,
        environment=ENVIRONMENT,
    ).inc()
