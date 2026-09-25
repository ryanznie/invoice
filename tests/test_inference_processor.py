"""
Tests for LayoutLMv3 processor loading hardening.
"""

from unittest.mock import Mock, patch

from src import inference


def test_processor_candidates_deduplicates_and_prefers_model_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_file = model_dir / "model.onnx"
    model_file.write_text("placeholder")

    candidates = inference._processor_candidates(str(model_file), str(model_dir))

    assert candidates[:4] == [
        str(model_dir),
        "models/artifacts",
        "models/layoutlmv3-lora-invoice-number",
        inference.BASE_MODEL,
    ]
    assert len(candidates) == len(set(candidates))


def test_build_layoutlmv3_processor_uses_manual_assembly_for_local_split_config(
    tmp_path,
):
    processor_dir = tmp_path / "processor"
    processor_dir.mkdir()
    expected_processor = Mock()
    image_processor = Mock()
    tokenizer = Mock()
    processor_cls = Mock(return_value=expected_processor)
    processor_cls.from_pretrained.side_effect = ValueError("missing processor config")

    with (
        patch("src.inference.LayoutLMv3Processor", processor_cls),
        patch(
            "src.inference.LayoutLMv3ImageProcessor.from_pretrained",
            return_value=image_processor,
        ) as image_from_pretrained,
        patch(
            "src.inference.LayoutLMv3TokenizerFast.from_pretrained",
            return_value=tokenizer,
        ) as tokenizer_from_pretrained,
    ):
        processor = inference._build_layoutlmv3_processor(str(processor_dir))

    assert processor is expected_processor
    processor_cls.from_pretrained.assert_called_once_with(
        str(processor_dir), apply_ocr=False
    )
    image_from_pretrained.assert_called_once_with(str(processor_dir), apply_ocr=False)
    tokenizer_from_pretrained.assert_called_once_with(str(processor_dir))
    processor_cls.assert_called_once_with(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )


def test_build_layoutlmv3_processor_reraises_for_non_directory_path():
    with patch(
        "src.inference.LayoutLMv3Processor.from_pretrained",
        side_effect=ValueError("not found"),
    ):
        try:
            inference._build_layoutlmv3_processor("missing/processor")
        except ValueError as exc:
            assert str(exc) == "not found"
        else:
            raise AssertionError("Expected processor load failure to be re-raised")
