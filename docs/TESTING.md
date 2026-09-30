# Testing

Install the development dependencies and run the suite from the repository root:

    uv sync --extra dev
    uv run pytest

Pytest discovers tests under tests/. CI runs the standard suite without tests/e2e, then runs the API contract check separately.

## API contract

This exercises the real FastAPI routes and multipart parser with local deterministic inference doubles. It does not call OpenRouter.

    uv run pytest --no-cov tests/e2e/test_api_contract.py -q

The result is written to docs/artifacts/api-contract-e2e.json. The covered failures and expected responses are listed in [API_CONTRACT_FAILURE_MATRIX.md](API_CONTRACT_FAILURE_MATRIX.md).

## Processor startup in the production image

Requires Docker and the pinned worker image. Hugging Face access is disabled inside the test container.

    RUN_PROCESSOR_CONTAINER_E2E=1 uv run pytest --no-cov tests/e2e/test_processor_startup.py -q

The result is written to docs/artifacts/processor-startup-e2e.json. This check is opt-in and is skipped by default.

## Other checks

Run the standalone fallback verification in its own process:

    uv run python tests/verify_fallback.py

Run the Locust profile against a live API. It requires the labeled dataset paths described in tests/load/locustfile.py:

    uv run locust -f tests/load/locustfile.py --host=http://localhost:7860

See [Monitoring](../monitoring/README.md) for saved load-test and offline-evaluation commands.
