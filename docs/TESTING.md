# Testing Guide

Production-ready test suite with 107 tests covering input validation, error handling, and integration workflows.

## Quick Start

```bash
# Install dependencies
uv sync

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov=scripts --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Suite

**107 tests** across 3 files:
- `tests/test_app.py` - Main application functions (59 tests)
- `tests/test_scripts.py` - Preprocessing utilities (35 tests)
- `tests/test_api.py` - API endpoints (13 tests)

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
