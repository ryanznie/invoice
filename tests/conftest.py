"""
Pytest configuration and shared fixtures for invoice NER tests
"""

import pytest
import json
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing"""
    # Create a simple 800x600 RGB image
    img_array = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGB")


@pytest.fixture
def small_image():
    """Create a small PIL Image for testing"""
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGB")


@pytest.fixture
def sample_words():
    """Sample OCR words"""
    return ["INVOICE", "NO:", "INV-12345", "DATE:", "2024-01-01"]


@pytest.fixture
def sample_boxes_normalized():
    """Sample bounding boxes in normalized format (0-1000)"""
    return [
        [100, 100, 200, 120],
        [210, 100, 280, 120],
        [290, 100, 400, 120],
        [100, 130, 180, 150],
        [190, 130, 300, 150],
    ]


@pytest.fixture
def sample_boxes_pixel():
    """Sample bounding boxes in pixel coordinates"""
    return [
        [80, 60, 160, 72],
        [168, 60, 224, 72],
        [232, 60, 320, 72],
        [80, 78, 144, 90],
        [152, 78, 240, 90],
    ]


@pytest.fixture
def sample_ocr_lines():
    """Sample OCR lines (before word splitting)"""
    return ["INVOICE NO: INV-12345", "DATE: 2024-01-01", "TOTAL: $100.00"]


@pytest.fixture
def temp_ocr_text_file():
    """Create a temporary OCR text file"""
    content = """83,41,331,41,331,78,83,78,TAN WOON YANN
109,171,330,171,330,191,109,191,MR D.I.Y. (M) SDN BHD
100,200,300,200,300,220,100,220,INVOICE NO: INV-12345
100,230,250,230,250,250,100,250,DATE: 2024-01-01"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file with OCR data"""
    data = {
        "words": ["INVOICE", "NO:", "INV-12345"],
        "bboxes": [[100, 100, 200, 120], [210, 100, 280, 120], [290, 100, 400, 120]],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def mock_model_output():
    """Mock model output structure"""
    return {
        "words": ["INVOICE", "NO:", "INV-12345"],
        "labels": ["LABEL_0", "LABEL_0", "LABEL_1"],
        "invoice_number": "INV-12345",
        "confidence_scores": [0.99, 0.98, 0.97],
    }


@pytest.fixture
def invoice_patterns():
    """Common invoice number patterns for testing"""
    return [
        ("INV# 12345", "12345"),
        ("INVOICE NO: ABC-123", "ABC-123"),
        ("INV-NO. TEST-456", "TEST-456"),
        ("RECEIPT NO 789-XYZ", "789-XYZ"),
        ("BILL NO: 2024-001", "2024-001"),
    ]
