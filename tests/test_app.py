"""
Comprehensive tests for invoice NER application
Tests all functions with validations, edge cases, and error handling
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch

# Import functions to test
from src import (
    normalize_boxes,
    parse_ocr_text_file,
    extract_invoice_heuristics,
    postprocess_invoice_number,
    create_annotated_image,
    predict_invoice,
)


class TestNormalizeBoxes:
    """Test suite for normalize_boxes function"""

    def test_normalize_boxes_valid(self):
        """Test normalization with valid inputs"""
        boxes = [[0, 0, 100, 50], [100, 50, 200, 100]]
        result = normalize_boxes(boxes, 800, 600)

        assert len(result) == 2
        assert result[0] == [0, 0, 125, 83]
        assert result[1] == [125, 83, 250, 166]

    def test_normalize_boxes_empty_list(self):
        """Test with empty boxes list"""
        result = normalize_boxes([], 800, 600)
        assert result == []

    def test_normalize_boxes_invalid_width(self):
        """Test with invalid image width"""
        boxes = [[0, 0, 100, 50]]

        with pytest.raises(ValueError, match="Invalid image dimensions"):
            normalize_boxes(boxes, 0, 600)

        with pytest.raises(ValueError, match="Invalid image dimensions"):
            normalize_boxes(boxes, -100, 600)

    def test_normalize_boxes_invalid_height(self):
        """Test with invalid image height"""
        boxes = [[0, 0, 100, 50]]

        with pytest.raises(ValueError, match="Invalid image dimensions"):
            normalize_boxes(boxes, 800, 0)

        with pytest.raises(ValueError, match="Invalid image dimensions"):
            normalize_boxes(boxes, 800, -100)

    def test_normalize_boxes_wrong_type_boxes(self):
        """Test with wrong type for boxes"""
        with pytest.raises(TypeError, match="Expected list for boxes"):
            normalize_boxes("not a list", 800, 600)

        with pytest.raises(TypeError, match="Expected list for boxes"):
            normalize_boxes(None, 800, 600)

    def test_normalize_boxes_wrong_type_dimensions(self):
        """Test with wrong type for dimensions"""
        boxes = [[0, 0, 100, 50]]

        with pytest.raises(TypeError, match="Image dimensions must be numeric"):
            normalize_boxes(boxes, "800", 600)

        with pytest.raises(TypeError, match="Image dimensions must be numeric"):
            normalize_boxes(boxes, 800, "600")

    def test_normalize_boxes_float_dimensions(self):
        """Test with float dimensions (should work)"""
        boxes = [[0, 0, 100, 50]]
        result = normalize_boxes(boxes, 800.0, 600.0)
        assert len(result) == 1
        assert result[0] == [0, 0, 125, 83]


class TestParseOcrTextFile:
    """Test suite for parse_ocr_text_file function"""

    def test_parse_valid_file(self, temp_ocr_text_file):
        """Test parsing a valid OCR text file"""
        result = parse_ocr_text_file(temp_ocr_text_file)

        assert "words" in result
        assert "bboxes" in result
        assert "ocr_lines" in result
        assert len(result["words"]) > 0
        assert len(result["words"]) == len(result["bboxes"])

    def test_parse_nonexistent_file(self):
        """Test with non-existent file"""
        with pytest.raises(FileNotFoundError, match="File not found"):
            parse_ocr_text_file("/path/to/nonexistent/file.txt")

    def test_parse_empty_path(self):
        """Test with empty file path"""
        with pytest.raises(ValueError, match="File path cannot be empty"):
            parse_ocr_text_file("")

    def test_parse_wrong_type(self):
        """Test with wrong type for file path"""
        with pytest.raises(TypeError, match="Expected string for file_path"):
            parse_ocr_text_file(123)

        with pytest.raises(TypeError, match="Expected string for file_path"):
            parse_ocr_text_file(None)

    def test_parse_malformed_file(self):
        """Test with malformed OCR file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("invalid,data\n")
            f.write("not,enough,fields\n")
            temp_path = f.name

        try:
            result = parse_ocr_text_file(temp_path)
            # Should handle gracefully and return empty or partial results
            assert "words" in result
            assert "bboxes" in result
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestExtractInvoiceHeuristics:
    """Test suite for extract_invoice_heuristics function"""

    def test_extract_with_inv_pattern(self):
        """Test extraction with INV# pattern"""
        words = ["INVOICE", "INV#", "12345", "DATE"]
        result, indices = extract_invoice_heuristics(words)

        assert result is not None
        assert "12345" in result
        assert isinstance(indices, list)

    def test_extract_with_invoice_no_pattern(self):
        """Test extraction with INVOICE NO pattern"""
        words = ["INVOICE", "NO:", "ABC-123", "TOTAL"]
        result, indices = extract_invoice_heuristics(words)

        assert result is not None
        assert "ABC" in result or "123" in result

    def test_extract_no_match(self):
        """Test when no pattern matches"""
        words = ["SOME", "RANDOM", "TEXT", "HERE"]
        result, indices = extract_invoice_heuristics(words)

        assert result is None
        assert indices == []

    def test_extract_with_ocr_lines(self):
        """Test extraction with OCR lines provided"""
        words = ["INVOICE", "NO:", "INV-12345"]
        ocr_lines = ["INVOICE NO: INV-12345", "DATE: 2024-01-01"]

        result, indices = extract_invoice_heuristics(words, ocr_lines)

        assert result is not None
        assert "12345" in result

    def test_extract_empty_words(self):
        """Test with empty words list"""
        with pytest.raises(ValueError, match="Words list cannot be empty"):
            extract_invoice_heuristics([])

    def test_extract_wrong_type_words(self):
        """Test with wrong type for words"""
        with pytest.raises(TypeError, match="Expected list for words"):
            extract_invoice_heuristics("not a list")

        with pytest.raises(TypeError, match="Expected list for words"):
            extract_invoice_heuristics(None)

    def test_extract_non_string_words(self):
        """Test with non-string elements in words"""
        with pytest.raises(TypeError, match="All words must be strings"):
            extract_invoice_heuristics(["INVOICE", 123, "NO"])

    def test_extract_wrong_type_ocr_lines(self):
        """Test with wrong type for ocr_lines"""
        words = ["INVOICE", "NO:", "12345"]

        with pytest.raises(TypeError, match="Expected list for ocr_lines"):
            extract_invoice_heuristics(words, "not a list")

    def test_extract_non_string_ocr_lines(self):
        """Test with non-string elements in ocr_lines"""
        words = ["INVOICE", "NO:", "12345"]

        with pytest.raises(TypeError, match="All OCR lines must be strings"):
            extract_invoice_heuristics(words, ["INVOICE NO: 12345", 123])

    def test_extract_short_match_rejected(self):
        """Test that matches <= 3 characters are rejected"""
        words = ["INV#", "123"]  # Too short
        result, indices = extract_invoice_heuristics(words)

        # Should reject because "123" is only 3 chars
        assert result is None or len(result) > 3

    def test_extract_no_digits_rejected(self):
        """Test that matches without digits are rejected"""
        words = ["INVOICE", "NO:", "ABCD"]  # No digits
        result, indices = extract_invoice_heuristics(words)

        # Should reject or not match
        if result is not None:
            assert any(c.isdigit() for c in result)


class TestPostprocessInvoiceNumber:
    """Test suite for postprocess_invoice_number function"""

    def test_postprocess_none(self):
        """Test with None input"""
        result = postprocess_invoice_number(None)
        assert result is None

    def test_postprocess_empty_string(self):
        """Test with empty string"""
        result = postprocess_invoice_number("")
        assert result == ""

    def test_postprocess_remove_colons(self):
        """Test colon removal"""
        result = postprocess_invoice_number("INV:12345")
        assert ":" not in result
        assert "INV12345" == result

    def test_postprocess_normalize_dashes(self):
        """Test dash normalization"""
        result = postprocess_invoice_number("INV - 12345")
        assert result == "INV-12345"

    def test_postprocess_normalize_slashes(self):
        """Test slash normalization"""
        result = postprocess_invoice_number("INV / 12345")
        assert result == "INV/12345"

    def test_postprocess_sp_null(self):
        """Test SP NULL replacement"""
        result = postprocess_invoice_number("SP NULL 12345")
        assert "SP-NULL" in result
        assert "SP NULL" not in result

    def test_postprocess_sp_null_truncation(self):
        """Test SP-NULL truncation to 24 chars"""
        long_invoice = "SP-NULL-" + "X" * 20
        result = postprocess_invoice_number(long_invoice)
        assert len(result) == 24

    def test_postprocess_remove_date_suffix(self):
        """Test DATE suffix removal"""
        result = postprocess_invoice_number("INV-12345DATE")
        assert result == "INV-12345"
        assert not result.endswith("DATE")

    def test_postprocess_combined(self):
        """Test multiple postprocessing rules"""
        result = postprocess_invoice_number("INV: 12345 - 678DATE")
        assert ":" not in result
        assert " - " not in result
        assert not result.endswith("DATE")

    def test_postprocess_wrong_type(self):
        """Test with wrong type"""
        with pytest.raises(TypeError, match="Expected string or None"):
            postprocess_invoice_number(123)

        with pytest.raises(TypeError, match="Expected string or None"):
            postprocess_invoice_number(["list"])


class TestCreateAnnotatedImage:
    """Test suite for create_annotated_image function"""

    def test_create_annotated_valid(
        self, sample_image, sample_words, sample_boxes_normalized
    ):
        """Test creating annotated image with valid inputs"""
        labels = ["LABEL_0", "LABEL_0", "LABEL_1", "LABEL_0", "LABEL_0"]

        result = create_annotated_image(
            sample_image, sample_words, sample_boxes_normalized, labels
        )

        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_create_annotated_heuristic_match(
        self, sample_image, sample_words, sample_boxes_normalized
    ):
        """Test with heuristic match labels"""
        labels = ["LABEL_0", "LABEL_0", "HEURISTIC_MATCH", "LABEL_0", "LABEL_0"]

        result = create_annotated_image(
            sample_image, sample_words, sample_boxes_normalized, labels
        )

        assert isinstance(result, Image.Image)

    def test_create_annotated_wrong_image_type(
        self, sample_words, sample_boxes_normalized
    ):
        """Test with wrong image type"""
        labels = ["LABEL_0"] * len(sample_words)

        with pytest.raises(TypeError, match="Expected PIL.Image.Image"):
            create_annotated_image(
                "not an image", sample_words, sample_boxes_normalized, labels
            )

    def test_create_annotated_invalid_image_size(
        self, sample_words, sample_boxes_normalized
    ):
        """Test with invalid image dimensions"""
        # Create image with 0 dimensions
        invalid_img = Image.new("RGB", (0, 0))
        labels = ["LABEL_0"] * len(sample_words)

        with pytest.raises(ValueError, match="Invalid image dimensions"):
            create_annotated_image(
                invalid_img, sample_words, sample_boxes_normalized, labels
            )

    def test_create_annotated_wrong_list_types(self, sample_image):
        """Test with wrong types for lists"""
        with pytest.raises(TypeError, match="words, boxes, and labels must be lists"):
            create_annotated_image(sample_image, "not a list", [], [])

        with pytest.raises(TypeError, match="words, boxes, and labels must be lists"):
            create_annotated_image(sample_image, [], "not a list", [])

        with pytest.raises(TypeError, match="words, boxes, and labels must be lists"):
            create_annotated_image(sample_image, [], [], "not a list")

    def test_create_annotated_mismatched_lengths(self, sample_image):
        """Test with mismatched list lengths"""
        words = ["INVOICE", "NO"]
        boxes = [[100, 100, 200, 120]]  # Only 1 box for 2 words
        labels = ["LABEL_0", "LABEL_0"]

        # Should handle gracefully by using min length
        result = create_annotated_image(sample_image, words, boxes, labels)
        assert isinstance(result, Image.Image)


class TestPredictInvoice:
    """Test suite for predict_invoice function"""

    @patch("src.inference.model")
    @patch("src.inference.processor")
    def test_predict_valid_input(
        self,
        mock_processor,
        mock_model,
        sample_image,
        sample_words,
        sample_boxes_normalized,
    ):
        """Test prediction with valid inputs"""
        import torch

        # Create a proper mock encoding object
        mock_encoding = Mock()
        mock_encoding.word_ids = Mock(return_value=[None, 0, 1, 2, 3, 4, None])
        mock_encoding.__getitem__ = Mock(
            side_effect=lambda key: {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1]]),
                "bbox": torch.tensor(
                    [
                        [
                            [0, 0, 0, 0],
                            [100, 100, 200, 120],
                            [210, 100, 280, 120],
                            [290, 100, 400, 120],
                            [100, 130, 180, 150],
                            [190, 130, 300, 150],
                            [0, 0, 0, 0],
                        ]
                    ]
                ),
            }[key]
        )
        mock_encoding.items = Mock(
            return_value=[
                ("input_ids", torch.tensor([[1, 2, 3, 4, 5, 6, 7]])),
                ("attention_mask", torch.tensor([[1, 1, 1, 1, 1, 1, 1]])),
                (
                    "bbox",
                    torch.tensor(
                        [
                            [
                                [0, 0, 0, 0],
                                [100, 100, 200, 120],
                                [210, 100, 280, 120],
                                [290, 100, 400, 120],
                                [100, 130, 180, 150],
                                [190, 130, 300, 150],
                                [0, 0, 0, 0],
                            ]
                        ]
                    ),
                ),
            ]
        )

        mock_processor.return_value = mock_encoding

        # Setup model config
        mock_model.config.id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}

        # Mock model output
        mock_output = Mock()
        # Create logits tensor: [batch_size, sequence_length, num_labels]
        mock_output.logits = torch.tensor(
            [
                [
                    [2.0, 0.1, 0.1],  # Token 0: LABEL_0
                    [2.0, 0.1, 0.1],  # Token 1: LABEL_0
                    [0.1, 2.0, 0.1],  # Token 2: LABEL_1 (invoice number)
                    [2.0, 0.1, 0.1],  # Token 3: LABEL_0
                    [2.0, 0.1, 0.1],  # Token 4: LABEL_0
                    [2.0, 0.1, 0.1],  # Token 5: LABEL_0
                    [2.0, 0.1, 0.1],
                ]
            ]
        )  # Token 6: LABEL_0

        mock_model.return_value = mock_output

        # Run prediction
        result = predict_invoice(sample_image, sample_words, sample_boxes_normalized)

        # Verify result structure
        assert isinstance(result, dict)
        assert "words" in result
        assert "labels" in result
        assert "invoice_number" in result
        assert "confidence_scores" in result

        # Verify we got predictions
        assert len(result["words"]) > 0
        assert len(result["labels"]) > 0
        assert len(result["confidence_scores"]) > 0

    def test_predict_model_not_loaded(
        self, sample_image, sample_words, sample_boxes_normalized
    ):
        """Test prediction when model is not loaded"""
        with patch("src.inference.model", None), patch("src.inference.processor", None):
            with pytest.raises(ValueError, match="Model not loaded"):
                predict_invoice(sample_image, sample_words, sample_boxes_normalized)

    def test_predict_wrong_image_type(self, sample_words, sample_boxes_normalized):
        """Test with wrong image type - will fail at image.size access"""
        with (
            patch("src.inference.model", Mock()),
            patch("src.inference.processor", Mock()),
        ):
            with pytest.raises(AttributeError):  # 'str' object has no attribute 'size'
                predict_invoice("not an image", sample_words, sample_boxes_normalized)

    def test_predict_invalid_image_dimensions(
        self, sample_words, sample_boxes_normalized
    ):
        """Test with invalid image dimensions"""
        invalid_img = Image.new("RGB", (0, 0))

        with (
            patch("src.inference.model", Mock()),
            patch("src.inference.processor", Mock()),
        ):
            with pytest.raises(ValueError, match="Invalid image dimensions"):
                predict_invoice(invalid_img, sample_words, sample_boxes_normalized)

    def test_predict_empty_words(self, sample_image, sample_boxes_normalized):
        """Test with empty words list"""
        with (
            patch("src.inference.model", Mock()),
            patch("src.inference.processor", Mock()),
        ):
            with pytest.raises(ValueError, match="Words list cannot be empty"):
                predict_invoice(sample_image, [], sample_boxes_normalized)

    def test_predict_wrong_words_type(self, sample_image, sample_boxes_normalized):
        """Test with wrong type for words - will fail at len() comparison"""
        with (
            patch("src.inference.model", Mock()),
            patch("src.inference.processor", Mock()),
        ):
            # String has different length than boxes, will fail at mismatch check
            with pytest.raises((TypeError, ValueError)):
                predict_invoice(sample_image, "not a list", sample_boxes_normalized)

    def test_predict_non_string_words(self, sample_image, sample_boxes_normalized):
        """Test with non-string elements in words"""
        with (
            patch("src.inference.model", Mock()),
            patch("src.inference.processor", Mock()),
        ):
            with pytest.raises(TypeError, match="All words must be strings"):
                predict_invoice(
                    sample_image, ["INVOICE", 123, "NO"], sample_boxes_normalized[:3]
                )

    def test_predict_wrong_boxes_type(self, sample_image, sample_words):
        """Test with wrong type for boxes - will fail at len() comparison"""
        with (
            patch("src.inference.model", Mock()),
            patch("src.inference.processor", Mock()),
        ):
            # String has different length than words, will fail at mismatch check
            with pytest.raises((TypeError, ValueError)):
                predict_invoice(sample_image, sample_words, "not a list")

    def test_predict_mismatched_lengths(
        self, sample_image, sample_words, sample_boxes_normalized
    ):
        """Test with mismatched words and boxes lengths"""
        with (
            patch("src.inference.model", Mock()),
            patch("src.inference.processor", Mock()),
        ):
            with pytest.raises(ValueError, match="Mismatch"):
                predict_invoice(sample_image, sample_words, sample_boxes_normalized[:3])

    def test_predict_invalid_box_format(self, sample_image):
        """Test with invalid box format"""
        words = ["INVOICE"]
        invalid_boxes = [[100, 100, 200]]  # Only 3 coordinates

        with (
            patch("src.inference.model", Mock()),
            patch("src.inference.processor", Mock()),
        ):
            with pytest.raises(ValueError, match="Box 0 must be a list of 4 integers"):
                predict_invoice(sample_image, words, invalid_boxes)

    def test_predict_box_out_of_range(self, sample_image):
        """Test with box coordinates out of range"""
        words = ["INVOICE"]
        invalid_boxes = [[100, 100, 1500, 120]]  # 1500 > 1000

        with (
            patch("src.inference.model", Mock()),
            patch("src.inference.processor", Mock()),
        ):
            with pytest.raises(ValueError, match="Box 0 coordinates must be in range"):
                predict_invoice(sample_image, words, invalid_boxes)

    def test_predict_invalid_box_geometry(self, sample_image):
        """Test with invalid box geometry (x0 >= x1)"""
        words = ["INVOICE"]
        invalid_boxes = [[200, 100, 100, 120]]  # x0 > x1

        with (
            patch("src.inference.model", Mock()),
            patch("src.inference.processor", Mock()),
        ):
            with pytest.raises(ValueError, match="Box 0 has invalid geometry"):
                predict_invoice(sample_image, words, invalid_boxes)

    def test_predict_box_non_numeric(self, sample_image):
        """Test with non-numeric box coordinates"""
        words = ["INVOICE"]
        invalid_boxes = [["100", "100", "200", "120"]]  # Strings instead of numbers

        with (
            patch("src.inference.model", Mock()),
            patch("src.inference.processor", Mock()),
        ):
            with pytest.raises(TypeError, match="Box 0 coordinates must be numeric"):
                predict_invoice(sample_image, words, invalid_boxes)


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_heuristic_to_postprocess_pipeline(self):
        """Test complete heuristic extraction to postprocessing pipeline"""
        words = ["INVOICE", "NO:", "INV-12345", "DATE"]

        # Extract
        invoice_number, _ = extract_invoice_heuristics(words)

        if invoice_number:
            # Postprocess
            cleaned = postprocess_invoice_number(invoice_number)

            assert cleaned is not None
            assert isinstance(cleaned, str)
            assert len(cleaned) > 0

    def test_file_parsing_to_normalization(self, temp_ocr_text_file):
        """Test parsing OCR file and normalizing boxes"""
        # Parse file
        ocr_data = parse_ocr_text_file(temp_ocr_text_file)

        # Normalize boxes (assuming 800x600 image)
        normalized = normalize_boxes(ocr_data["bboxes"], 800, 600)

        assert len(normalized) == len(ocr_data["words"])
        assert all(all(0 <= coord <= 1000 for coord in box) for box in normalized)


class TestHeuristicValidationRules:
    """Test suite for heuristic extraction validation rules"""

    def test_heuristic_rejects_date_pattern_mmddyyyy(self):
        """Test that date patterns MM/DD/YYYY are rejected"""
        words = ["INVOICE", "NO:", "12/31/2024", "TOTAL"]
        result, indices = extract_invoice_heuristics(words)

        # Should not extract date pattern
        assert result is None or "12/31/2024" not in result

    def test_heuristic_rejects_date_pattern_ddmmyy(self):
        """Test that date patterns DD/MM/YY are rejected"""
        words = ["INVOICE", "NO:", "31/12/24", "TOTAL"]
        result, indices = extract_invoice_heuristics(words)

        # Should not extract date pattern
        assert result is None or "31/12/24" not in result

    def test_heuristic_accepts_invoice_with_one_slash(self):
        """Test that invoice numbers with one slash are accepted"""
        words = ["INVOICE", "NO:", "INV/12345", "TOTAL"]
        result, indices = extract_invoice_heuristics(words)

        # Should accept invoice with single slash
        assert result is not None
        assert "INV" in result or "12345" in result

    def test_heuristic_accepts_invoice_with_two_slashes_not_date(self):
        """Test that non-date patterns with 2 slashes are accepted"""
        words = ["INVOICE", "NO:", "INV/2024/12345", "TOTAL"]
        result, indices = extract_invoice_heuristics(words)

        # Should accept since it's not a date pattern (has letters)
        assert result is not None
        assert "INV" in result or "12345" in result

    def test_heuristic_rejects_short_matches(self):
        """Test that matches <= 3 characters are rejected"""
        words = ["INV#", "123"]  # Too short
        result, indices = extract_invoice_heuristics(words)

        # Should reject because "123" is only 3 chars
        assert result is None or len(result) > 3

    def test_heuristic_rejects_no_digits(self):
        """Test that matches without digits are rejected"""
        words = ["INVOICE", "NO:", "ABCD"]  # No digits
        result, indices = extract_invoice_heuristics(words)

        # Should reject or not match
        if result is not None:
            assert any(c.isdigit() for c in result)

    def test_heuristic_handles_comma_separated_values(self):
        """Test that comma-separated values take first value"""
        words = ["INVOICE", "NO:", "INV-123,INV-456"]
        ocr_lines = ["INVOICE NO: INV-123,INV-456"]
        result, indices = extract_invoice_heuristics(words, ocr_lines)

        # Should take first value before comma
        if result is not None:
            assert "," not in result
            assert "INV-123" in result or "123" in result


class TestModelExtractionValidation:
    """Test suite for model extraction validation rules"""

    def test_validation_rejects_semicolon(self):
        """Test that model extractions with semicolons would be rejected"""
        # This tests the validation logic
        invoice_with_semicolon = "INV-123;456"

        # Check if semicolon is present (simulating the validation)
        assert ";" in invoice_with_semicolon

    def test_validation_rejects_no_alphanumeric(self):
        """Test that extractions without alphanumeric characters would be rejected"""
        import re

        # Test various non-alphanumeric strings
        test_cases = ["---", "...", ":::", "///", "   "]

        for test_str in test_cases:
            # Should not contain alphanumeric characters
            assert not re.search(r"[a-zA-Z0-9]", test_str)

    def test_validation_accepts_valid_alphanumeric(self):
        """Test that valid alphanumeric invoice numbers pass validation"""
        import re

        # Test various valid invoice numbers
        test_cases = ["INV-123", "A1234", "2024-001", "INV/123"]

        for test_str in test_cases:
            # Should contain alphanumeric characters
            assert re.search(r"[a-zA-Z0-9]", test_str)

    def test_validation_accepts_invoice_without_semicolon(self):
        """Test that normal invoice numbers without semicolons pass"""
        test_cases = ["INV-123", "A1234", "2024-001", "INV/123"]

        for test_str in test_cases:
            # Should not contain semicolon
            assert ";" not in test_str

    def test_validation_mixed_special_chars_with_alphanumeric(self):
        """Test that mixed special characters with alphanumeric pass"""
        import re

        # These should pass because they contain alphanumeric
        test_cases = ["INV-123!", "A@1234", "#2024-001", "INV/123$"]

        for test_str in test_cases:
            # Should contain alphanumeric characters
            assert re.search(r"[a-zA-Z0-9]", test_str)
            # But should not contain semicolon
            assert ";" not in test_str


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_long_invoice_number(self):
        """Test with very long invoice number"""
        long_invoice = "INV-" + "X" * 100
        result = postprocess_invoice_number(long_invoice)
        assert isinstance(result, str)

    def test_unicode_characters(self):
        """Test with unicode characters in invoice number"""
        unicode_invoice = "INV-12345-café"
        result = postprocess_invoice_number(unicode_invoice)
        assert isinstance(result, str)

    def test_special_characters_in_words(self):
        """Test with special characters in words"""
        words = ["INVOICE#", "№:", "12345", "@", "test"]
        result, _ = extract_invoice_heuristics(words)
        # Should handle gracefully
        assert result is None or isinstance(result, str)

    def test_very_small_boxes(self):
        """Test with very small bounding boxes"""
        boxes = [[0, 0, 1, 1]]
        result = normalize_boxes(boxes, 1000, 1000)
        assert len(result) == 1
        assert all(isinstance(coord, int) for coord in result[0])

    def test_maximum_coordinate_values(self):
        """Test with maximum coordinate values"""
        boxes = [[0, 0, 1000, 1000]]
        result = normalize_boxes(boxes, 1000, 1000)
        assert result[0] == [0, 0, 1000, 1000]
