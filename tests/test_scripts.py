"""
Tests for preprocessing scripts
"""

from scripts.preprocess import (
    split_invoice_string,
    estimate_word_boxes,
    normalize_bbox,
)


class TestSplitInvoiceString:
    """Test suite for split_invoice_string function"""

    def test_split_simple_string(self):
        """Test splitting simple invoice string"""
        result = split_invoice_string("INV-12345")
        assert "INV" in result
        assert "-" in result
        assert "12345" in result

    def test_split_with_slash(self):
        """Test splitting with slash delimiter"""
        result = split_invoice_string("INV/2024/001")
        assert "INV" in result
        assert "/" in result
        assert "2024" in result
        assert "001" in result

    def test_split_with_colon(self):
        """Test splitting with colon"""
        result = split_invoice_string("INV:12345")
        assert "INV" in result
        assert ":" in result
        assert "12345" in result

    def test_split_with_hash(self):
        """Test splitting with hash"""
        result = split_invoice_string("INV#12345")
        assert "INV" in result
        assert "#" in result
        assert "12345" in result

    def test_split_with_parentheses(self):
        """Test splitting with parentheses"""
        result = split_invoice_string("INV(2024)")
        assert "INV" in result
        assert "(" in result
        assert "2024" in result
        assert ")" in result

    def test_split_with_brackets(self):
        """Test splitting with brackets"""
        result = split_invoice_string("INV[123]")
        assert "INV" in result
        assert "[" in result
        assert "123" in result
        assert "]" in result

    def test_split_with_dot(self):
        """Test splitting with dot"""
        result = split_invoice_string("INV.12345")
        assert "INV" in result
        assert "." in result
        assert "12345" in result

    def test_split_with_spaces(self):
        """Test splitting with spaces"""
        result = split_invoice_string("INV 12345 ABC")
        assert "INV" in result
        assert "12345" in result
        assert "ABC" in result

    def test_split_complex_invoice(self):
        """Test splitting complex invoice number"""
        result = split_invoice_string("INV-2024/001:ABC")
        assert "INV" in result
        assert "-" in result
        assert "2024" in result
        assert "/" in result
        assert "001" in result
        assert ":" in result
        assert "ABC" in result

    def test_split_empty_string(self):
        """Test with empty string"""
        result = split_invoice_string("")
        assert result == []

    def test_split_whitespace_only(self):
        """Test with whitespace only"""
        result = split_invoice_string("   ")
        assert result == []

    def test_split_preserves_delimiters(self):
        """Test that delimiters are preserved"""
        result = split_invoice_string("A-B/C:D")
        assert "-" in result
        assert "/" in result
        assert ":" in result

    def test_split_no_delimiters(self):
        """Test string with no delimiters"""
        result = split_invoice_string("INV12345")
        assert result == ["INV12345"]

    def test_split_multiple_spaces(self):
        """Test with multiple consecutive spaces"""
        result = split_invoice_string("INV    12345")
        assert "INV" in result
        assert "12345" in result
        # Should not have empty strings
        assert "" not in result


class TestEstimateWordBoxes:
    """Test suite for estimate_word_boxes function"""

    def test_estimate_single_token(self):
        """Test with single token"""
        line_text = "INVOICE"
        tokens = ["INVOICE"]
        line_bbox = [100, 50, 200, 70]

        result = estimate_word_boxes(line_text, tokens, line_bbox)

        assert len(result) == 1
        assert result[0] == line_bbox

    def test_estimate_multiple_tokens(self):
        """Test with multiple tokens"""
        line_text = "INVOICE NO 12345"
        tokens = ["INVOICE", "NO", "12345"]
        line_bbox = [100, 50, 400, 70]

        result = estimate_word_boxes(line_text, tokens, line_bbox)

        assert len(result) == 3
        # All boxes should have same y coordinates
        assert all(box[1] == 50 for box in result)
        assert all(box[3] == 70 for box in result)
        # Boxes should be in order
        assert result[0][0] < result[1][0] < result[2][0]

    def test_estimate_proportional_distribution(self):
        """Test that boxes are distributed proportionally"""
        line_text = "AB CDEF"
        tokens = ["AB", "CDEF"]
        line_bbox = [0, 0, 100, 20]

        result = estimate_word_boxes(line_text, tokens, line_bbox)

        assert len(result) == 2
        # First token "AB" should take less space than "CDEF"
        width1 = result[0][2] - result[0][0]
        width2 = result[1][2] - result[1][0]
        assert width2 > width1

    def test_estimate_token_not_found_fallback(self):
        """Test fallback when token not found in text"""
        line_text = "INVOICE"
        tokens = ["XYZ", "ABC"]  # Tokens not in text
        line_bbox = [100, 50, 200, 70]

        result = estimate_word_boxes(line_text, tokens, line_bbox)

        # Should use equal distribution fallback
        assert len(result) == 2
        width1 = result[0][2] - result[0][0]
        width2 = result[1][2] - result[1][0]
        # Should be roughly equal
        assert abs(width1 - width2) < 5

    def test_estimate_preserves_y_coordinates(self):
        """Test that y coordinates are preserved"""
        line_text = "A B C"
        tokens = ["A", "B", "C"]
        line_bbox = [100, 200, 400, 250]

        result = estimate_word_boxes(line_text, tokens, line_bbox)

        assert all(box[1] == 200 for box in result)
        assert all(box[3] == 250 for box in result)

    def test_estimate_x_boundaries(self):
        """Test that x coordinates stay within line boundaries"""
        line_text = "INVOICE NO"
        tokens = ["INVOICE", "NO"]
        line_bbox = [100, 50, 300, 70]

        result = estimate_word_boxes(line_text, tokens, line_bbox)

        # First box should start at line start
        assert result[0][0] >= 100
        # Last box should end at or before line end
        assert result[-1][2] <= 300


class TestNormalizeBbox:
    """Test suite for normalize_bbox function"""

    def test_normalize_center_box(self):
        """Test normalizing a centered box"""
        bbox = [400, 300, 600, 450]
        result = normalize_bbox(bbox, 800, 600)

        assert result == [500, 500, 750, 750]

    def test_normalize_top_left_corner(self):
        """Test normalizing box at top-left corner"""
        bbox = [0, 0, 100, 50]
        result = normalize_bbox(bbox, 800, 600)

        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 125
        assert result[3] == 83

    def test_normalize_bottom_right_corner(self):
        """Test normalizing box at bottom-right corner"""
        bbox = [700, 550, 800, 600]
        result = normalize_bbox(bbox, 800, 600)

        assert result[2] == 1000
        assert result[3] == 1000

    def test_normalize_full_image(self):
        """Test normalizing box covering full image"""
        bbox = [0, 0, 800, 600]
        result = normalize_bbox(bbox, 800, 600)

        assert result == [0, 0, 1000, 1000]

    def test_normalize_small_box(self):
        """Test normalizing very small box"""
        bbox = [100, 100, 101, 101]
        result = normalize_bbox(bbox, 1000, 1000)

        assert len(result) == 4
        assert all(isinstance(coord, int) for coord in result)

    def test_normalize_different_aspect_ratio(self):
        """Test with different aspect ratios"""
        # Wide image
        bbox = [0, 0, 1600, 600]
        result = normalize_bbox(bbox, 1600, 600)
        assert result == [0, 0, 1000, 1000]

        # Tall image
        bbox = [0, 0, 600, 1600]
        result = normalize_bbox(bbox, 600, 1600)
        assert result == [0, 0, 1000, 1000]

    def test_normalize_returns_integers(self):
        """Test that result contains only integers"""
        bbox = [123, 456, 789, 543]
        result = normalize_bbox(bbox, 800, 600)

        assert all(isinstance(coord, int) for coord in result)

    def test_normalize_maintains_order(self):
        """Test that normalized box maintains x0 < x1 and y0 < y1"""
        bbox = [100, 100, 200, 200]
        result = normalize_bbox(bbox, 800, 600)

        assert result[0] < result[2]  # x0 < x1
        assert result[1] < result[3]  # y0 < y1


class TestEdgeCases:
    """Test edge cases for preprocessing functions"""

    def test_split_unicode_characters(self):
        """Test splitting with unicode characters"""
        result = split_invoice_string("INV-café-123")
        assert "INV" in result
        assert "café" in result
        assert "123" in result

    def test_estimate_empty_tokens(self):
        """Test estimate_word_boxes with empty tokens list"""
        result = estimate_word_boxes("text", [], [0, 0, 100, 20])
        assert result == []

    def test_normalize_zero_width_box(self):
        """Test normalizing box with zero width"""
        bbox = [100, 100, 100, 200]
        result = normalize_bbox(bbox, 800, 600)
        # Should still work, though box has no width
        assert len(result) == 4

    def test_split_consecutive_delimiters(self):
        """Test with consecutive delimiters"""
        result = split_invoice_string("INV--//123")
        assert "INV" in result
        # Should preserve delimiters
        assert result.count("-") == 2
        assert result.count("/") == 2

    def test_estimate_very_long_text(self):
        """Test with very long text"""
        line_text = "A" * 1000
        tokens = ["A" * 500, "A" * 500]
        line_bbox = [0, 0, 1000, 20]

        result = estimate_word_boxes(line_text, tokens, line_bbox)
        assert len(result) == 2

    def test_normalize_large_coordinates(self):
        """Test with very large image dimensions"""
        bbox = [0, 0, 10000, 10000]
        result = normalize_bbox(bbox, 10000, 10000)
        assert result == [0, 0, 1000, 1000]
