# Tests

Run the pytest suite from the repository root:

    uv run pytest

Top-level test modules cover the app, API, preprocessing scripts, Runpod handler, monitoring, and robustness. Container and API contract checks are in tests/e2e/; the Locust profile is in tests/load/.

The fallback verification script is not auto-discovered by pytest. Run it separately:

    uv run python tests/verify_fallback.py

See [docs/TESTING.md](../docs/TESTING.md) for E2E, CI, and load-test details.
