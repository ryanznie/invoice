# Testing Guide

The project favors end-to-end checks for behavioral changes. API contract runs
write deterministic JSON artifacts that can be inspected or compared in CI.

## Quick Start

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov=scripts --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Backend API Contract E2E

Run the request contract through the real FastAPI routes and multipart parser:

```bash
uv run pytest --no-cov tests/e2e/test_api_contract.py -q
```

This covers readiness, pre-readiness rejection, exact and oversized upload
boundaries, malformed requests, sanitized errors, and the enabled/disabled
OpenRouter fallback branches. It never calls OpenRouter. The deterministic
result is written to
[`artifacts/api-contract-e2e.json`](artifacts/api-contract-e2e.json).

Run the real bundled processor and ONNX model startup branches offline in the
pinned production image:

```bash
RUN_PROCESSOR_CONTAINER_E2E=1 \
  uv run pytest --no-cov tests/e2e/test_processor_startup.py -q
```

This checks both the bundled processor path and the fallback from an invalid
`PROCESSOR_PATH` to the local `BASE_MODEL`. `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE` are enabled inside the container, so a passing run proves
that startup does not depend on a Hugging Face download. The result is written
to
[`artifacts/processor-startup-e2e.json`](artifacts/processor-startup-e2e.json).

The complete failure inventory and expected responses are in
[`API_CONTRACT_FAILURE_MATRIX.md`](API_CONTRACT_FAILURE_MATRIX.md).

## Test Suite

The pytest suite covers existing application and preprocessing behavior. The
two executable E2E checks above own the new deployment API and processor-startup
contracts.

## What's Tested

All functions have comprehensive validation:
- ✅ Input types and ranges
- ✅ Error handling and edge cases
- ✅ Integration workflows
- ✅ API endpoints

## Running Specific Tests

```bash
# By file
pytest tests/test_app.py

# By class
pytest tests/test_app.py::TestPredictInvoice
pytest tests/test_scripts.py::TestSplitInvoiceString
```

### By Test Function
```bash
pytest tests/test_app.py::TestPredictInvoice::test_predict_invalid_box_geometry
```

### By Pattern
```bash
pytest -k "validation"             # Run tests with "validation" in name
pytest -k "edge_case"              # Run edge case tests
pytest -k "normalize"              # Run normalization tests
```

## CI/CD Integration

Tests run automatically on every push and pull request via GitHub Actions.

See `.github/workflows/ci.yml` for the full configuration.

## Troubleshooting

**Import errors**: Run from project root
```bash
cd /Users/ryanznie/Desktop/work/invoice-ner
pytest
```

**Model loading**: Tests mock the model by default

For more details, see `tests/README.md`
