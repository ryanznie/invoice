# Invoice NER Test Suite

Comprehensive test suite for the Invoice NER application using pytest.

## Overview

This test suite provides production-ready testing for:
- **Input validation** for all functions
- **Edge cases** and boundary conditions
- **Error handling** and exception raising
- **Integration tests** for complete workflows
- **API endpoint testing** with FastAPI TestClient

## Test Structure

```
tests/
├── __init__.py           # Package initialization
├── conftest.py           # Shared fixtures and configuration
├── test_app.py           # Tests for main application functions
├── test_scripts.py       # Tests for preprocessing scripts
├── test_api.py           # Tests for FastAPI endpoints
└── README.md             # This file
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_app.py
pytest tests/test_scripts.py
pytest tests/test_api.py
```

### Run Specific Test Class
```bash
pytest tests/test_app.py::TestNormalizeBoxes
pytest tests/test_app.py::TestPredictInvoice
```

### Run Specific Test Function
```bash
pytest tests/test_app.py::TestNormalizeBoxes::test_normalize_boxes_valid
```

### Run with Coverage Report
```bash
pytest --cov=app --cov=scripts --cov-report=html
```

Then open `htmlcov/index.html` in your browser.

### Run with Verbose Output
```bash
pytest -v
```

### Run Only Fast Tests (Skip Slow Tests)
```bash
pytest -m "not slow"
```

### Run Only Validation Tests
```bash
pytest -m validation
```

### Stop on First Failure
```bash
pytest -x
```

### Show Local Variables on Failure
```bash
pytest -l
```

## Test Categories

### Unit Tests
Test individual functions in isolation:
- `test_normalize_boxes_valid()`
- `test_extract_invoice_heuristics()`
- `test_postprocess_invoice_number()`

### Validation Tests
Test input validation and error handling:
- `test_predict_wrong_image_type()`
- `test_normalize_boxes_invalid_width()`
- `test_extract_empty_words()`

### Integration Tests
Test complete workflows:
- `test_heuristic_to_postprocess_pipeline()`
- `test_file_parsing_to_normalization()`

### Edge Case Tests
Test boundary conditions and special cases:
- `test_very_long_invoice_number()`
- `test_unicode_characters()`
- `test_maximum_coordinate_values()`

### API Tests
Test FastAPI endpoints:
- `test_health_check_model_loaded()`
- `test_root_endpoint_metadata()`
- `test_predict_returns_boxes_for_frontend()`

## Fixtures

### Image Fixtures
- `sample_image`: 800x600 RGB image
- `small_image`: 100x100 RGB image

### Data Fixtures
- `sample_words`: List of OCR words
- `sample_boxes_normalized`: Normalized bounding boxes (0-1000)
- `sample_boxes_pixel`: Pixel coordinate bounding boxes
- `sample_ocr_lines`: Original OCR text lines

### File Fixtures
- `temp_ocr_text_file`: Temporary OCR text file
- `temp_json_file`: Temporary JSON file with OCR data

### Mock Fixtures
- `mock_model_output`: Mock model prediction output
- `invoice_patterns`: Common invoice number patterns

## Coverage Goals

Target coverage: **>80%** for production code

Current coverage areas:
- ✅ Input validation functions
- ✅ Data preprocessing functions
- ✅ Heuristic extraction
- ✅ Postprocessing functions
- ✅ Image annotation
- ✅ API endpoints
- ⚠️ Model inference (requires model loading)

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test
```python
def test_function_name_scenario(fixture_name):
    """Test description"""
    # Arrange
    input_data = ...
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_value
```

### Testing Exceptions
```python
def test_function_raises_error():
    """Test that function raises appropriate error"""
    with pytest.raises(ValueError, match="error message"):
        function_that_should_fail(invalid_input)
```

### Using Fixtures
```python
def test_with_fixture(sample_image, sample_words):
    """Test using shared fixtures"""
    result = process_image(sample_image, sample_words)
    assert result is not None
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pytest --cov=app --cov=scripts --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Import Errors
If you get import errors, ensure you're running from the project root:
```bash
cd /path/to/invoice-ner
pytest
```

### Model Loading Issues
Some tests mock the model to avoid loading it. If you want to test with the actual model:
```bash
pytest -m "not requires_model"  # Skip model tests
```

### Fixture Not Found
Ensure `conftest.py` is in the tests directory and pytest can discover it.

## Best Practices

1. **Test one thing at a time**: Each test should verify one specific behavior
2. **Use descriptive names**: Test names should clearly indicate what they test
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Use fixtures**: Share common setup code via fixtures
5. **Test edge cases**: Don't just test the happy path
6. **Mock external dependencies**: Use mocks for models, APIs, file systems when appropriate
7. **Keep tests fast**: Unit tests should run in milliseconds
8. **Test error handling**: Verify that functions fail gracefully with appropriate errors

## Dependencies

Required packages (already in `pyproject.toml`):
- `pytest>=8.0.0`
- `pytest-cov>=4.1.0`

Optional but recommended:
- `pytest-xdist` - Run tests in parallel
- `pytest-timeout` - Timeout for long-running tests
- `pytest-mock` - Enhanced mocking capabilities

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain >80% code coverage
4. Add validation tests for all inputs
5. Test edge cases and error conditions
