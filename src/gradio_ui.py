"""
Gradio interface for Invoice NER demo.
"""

import json
import logging
from PIL import Image, ImageDraw
import gradio as gr

from .inference import predict_invoice
from .heuristics import extract_invoice_heuristics
from .postprocessing import postprocess_invoice_number
from .utils import parse_ocr_text_file, normalize_boxes

logger = logging.getLogger(__name__)


# ============================================================================
# GRADIO PREDICTION & VISUALIZATION
# ============================================================================


def gradio_predict(image, text_file):
    """
    Gradio prediction function

    Args:
        image: Uploaded image
        text_file: Uploaded text/JSON file with OCR data

    Returns:
        Formatted results string and highlighted image
    """
    if image is None:
        return "Please upload an image", None, ""

    if text_file is None:
        return "Please upload a text file with OCR data", None, ""

    try:
        # Convert image to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Get image dimensions
        img_width, img_height = image.size

        # Determine file type and parse accordingly
        file_path = text_file.name

        if file_path.endswith(".json"):
            # Read JSON file
            with open(file_path, "r") as f:
                ocr_data = json.load(f)

            # Extract words and boxes
            words = ocr_data.get("words", [])
            boxes = ocr_data.get("bboxes", ocr_data.get("boxes", []))
            ocr_lines = ocr_data.get("ocr_lines", None)  # May not be present in JSON

            # Check if boxes need normalization (if any coordinate > 1000, assume pixel coordinates)
            needs_normalization = (
                any(coord > 1000 for box in boxes for coord in box) if boxes else False
            )

            if needs_normalization:
                logger.info(
                    "Detected pixel coordinates in JSON, normalizing to 0-1000 range"
                )
                boxes = normalize_boxes(boxes, img_width, img_height)
        else:
            # Parse text file format
            ocr_data = parse_ocr_text_file(file_path)
            words = ocr_data["words"]
            boxes = ocr_data["bboxes"]
            ocr_lines = ocr_data.get("ocr_lines", None)

            # Normalize boxes to 0-1000 range
            boxes = normalize_boxes(boxes, img_width, img_height)

        if not words or not boxes:
            return (
                "File must contain valid OCR data",
                None,
                "",
            )

        logger.info("=" * 60)
        logger.info("Starting invoice extraction pipeline")
        logger.info(f"Total words: {len(words)}, Total boxes: {len(boxes)}")
        if ocr_lines:
            logger.info(f"OCR lines available: {len(ocr_lines)}")
        logger.info("=" * 60)

        # Step 1: Try heuristics first (with original OCR lines if available)
        invoice_number, matched_indices = extract_invoice_heuristics(words, ocr_lines)

        if invoice_number:
            # Heuristic found a match
            extraction_method = "🎯 Heuristic"
            logger.info(f"Using heuristic extraction result: '{invoice_number}'")
            # Create labels for visualization (green boxes for matched words)
            labels = [
                "HEURISTIC_MATCH" if i in matched_indices else "LABEL_0"
                for i in range(len(words))
            ]
            result = {
                "words": words,
                "labels": labels,
                "confidence_scores": [1.0] * len(words),
                "matched_indices": matched_indices,
            }
        else:
            # Step 2: Fall back to model
            extraction_method = "🤖 Model"
            logger.info("🤖 Falling back to LayoutLMv3 model inference")
            result = predict_invoice(image, words, boxes)
            invoice_number = result["invoice_number"]
            logger.info(f"Model extracted: '{invoice_number}'")

            # Step 3: Apply postprocessing to model results
            if invoice_number:
                invoice_number = postprocess_invoice_number(invoice_number)

        # Final display
        invoice_number = invoice_number or "Not Found"
        logger.info("=" * 60)
        logger.info(f"FINAL RESULT: '{invoice_number}' (via {extraction_method})")
        logger.info("=" * 60)

        # Create annotated image and detailed output
        if extraction_method == "🎯 Heuristic":
            # Heuristic extraction - show green boxes for matched words
            annotated_image = create_annotated_image(
                image, words, boxes, result["labels"]
            )
            detailed_output = f"## 📊 Extraction Method: {extraction_method}\n\n"
            detailed_output += (
                "Invoice number extracted using **pattern matching heuristics**.\n\n"
            )
            detailed_output += "✨ Fast extraction without model inference.\n\n"
            detailed_output += f"🔴 **{len(result['matched_indices'])} word(s)** matched the heuristic pattern.\n"

            # Warning if multiple words matched
            if len(result["matched_indices"]) >= 2:
                detailed_output += "\n⚠️ **WARNING**: Multiple words detected. Please inspect carefully to ensure accuracy.\n"
                logger.warning(
                    f"Multiple words ({len(result['matched_indices'])}) matched heuristic pattern"
                )
        else:
            # Model extraction - show annotations and predictions
            annotated_image = create_annotated_image(
                image, words, boxes, result["labels"]
            )

            # Count predicted invoice number words
            predicted_count = sum(
                1 for label in result["labels"] if label.startswith("LABEL_1")
            )

            detailed_output = f"## 📊 Extraction Method: {extraction_method}\n\n"
            detailed_output += "Invoice number extracted using LayoutLMv3 model.\n\n"

            # Warning if multiple predictions
            if predicted_count >= 2:
                detailed_output += "⚠️ **WARNING**: Multiple words predicted as invoice number. Please inspect carefully to ensure accuracy.\n\n"
                logger.warning(
                    f"Multiple words ({predicted_count}) predicted as invoice number by model"
                )

            detailed_output += "### 📋 Word-level Predictions\n\n"
            detailed_output += "| Word | Label | Confidence |\n"
            detailed_output += "|------|-------|------------|\n"

            for word, label, conf in zip(
                result["words"], result["labels"], result["confidence_scores"]
            ):
                emoji = (
                    "🟢"
                    if label.startswith("LABEL_1") or label.startswith("LABEL_2")
                    else "⚪"
                )
                detailed_output += f"| {emoji} {word} | `{label}` | {conf:.3f} |\n"

        return invoice_number, annotated_image, detailed_output

    except Exception as e:
        return f"❌ Error: {str(e)}", None, ""


def create_annotated_image(image, words, boxes, labels):
    """
    Create an annotated image with bounding boxes

    Args:
        image: PIL Image
        words: List of words
        boxes: List of bounding boxes
        labels: List of predicted labels (may be shorter than words/boxes)

    Returns:
        Annotated PIL Image

    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image).__name__}")

    if (
        not isinstance(words, list)
        or not isinstance(boxes, list)
        or not isinstance(labels, list)
    ):
        raise TypeError("words, boxes, and labels must be lists")

    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")

    # Create a copy
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Get image dimensions
    img_width, img_height = img.size

    # Handle case where labels list is shorter (e.g., heuristic extraction)
    num_to_annotate = min(len(words), len(boxes), len(labels))

    for i in range(num_to_annotate):
        # word = words[i]
        box = boxes[i]
        label = labels[i]

        # Denormalize coordinates (boxes are in 0-1000 range)
        x0 = int(box[0] * img_width / 1000)
        y0 = int(box[1] * img_height / 1000)
        x1 = int(box[2] * img_width / 1000)
        y1 = int(box[3] * img_height / 1000)

        # Choose color based on label
        if (
            label == "HEURISTIC_MATCH"
            or label.startswith("LABEL_1")
            or label.startswith("LABEL_2")
        ):
            # Red for both heuristic matches and model predictions
            color = "red"
            width_box = 3
        else:
            # Light blue for other words
            color = "lightblue"
            width_box = 1

        # Draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width_box)

    return img


# ============================================================================
# GRADIO UI SETUP
# ============================================================================


def create_gradio_interface():
    """
    Create and return the Gradio interface.

    Returns:
        gr.Blocks: Configured Gradio interface
    """
    with gr.Blocks(title="Invoice NER Demo") as demo:
        gr.Markdown("# 🧾 Invoice Number Extraction")

        # How to Use section at the top
        gr.Markdown("## 💡 How to Use")
        gr.Markdown(
            """
            1. Upload an invoice image (JPG, PNG)
            2. Upload OCR data:
               - **Text file (.txt)**: One line per word in format `x1,y1,x2,y2,x3,y3,x4,y4,text`
               - **JSON file (.json)**: With `words` and `bboxes` fields
            3. Click "Extract Invoice Number"
            
            **Hybrid Extraction:**
            - 🎯 **Heuristics First**: Tries pattern matching (e.g., "INV#", "INVOICE NO") for fast extraction
            - 🤖 **Model Fallback**: Uses LayoutLMv3 if no heuristic matches
            - ✨ **Postprocessing**: Cleans up results (removes extra spaces, colons, etc.)
            
            **Color Coding:**
            - 🔴 **Red**: Detected invoice numbers (heuristic or model)
            - 🔵 **Light Blue**: Other text
            """
        )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="📷 Upload Invoice Image", type="pil")
                text_input = gr.File(
                    label="📄 Upload OCR Data (TXT or JSON)",
                    file_types=[".txt", ".json"],
                )
                predict_btn = gr.Button("🚀 Extract Invoice Number", variant="primary")

            with gr.Column():
                # Large prominent display for invoice number
                invoice_output = gr.Textbox(
                    label="🎯 Extracted Invoice Number",
                    placeholder="Invoice number will appear here...",
                    interactive=False,
                    scale=2,
                    container=True,
                    show_label=True,
                )
                output_image = gr.Image(label="📊 Annotated Image")
                output_text = gr.Markdown(label="📋 Detailed Results")

        # Supported file formats info at bottom
        with gr.Accordion("📝 Supported File Formats", open=False):
            gr.Markdown(
                """
                **Option 1: Text File (.txt)**
                ```
                x1,y1,x2,y2,x3,y3,x4,y4,text
                83,41,331,41,331,78,83,78,TAN WOON YANN
                109,171,330,171,330,191,109,191,MR D.I.Y. (M) SDN BHD
                ```
                
                **Option 2: JSON File (.json)**
                ```json
                {
                    "words": ["INVOICE", "NO:", "INV-12345"],
                    "bboxes": [[100, 100, 200, 120], [210, 100, 280, 120], ...]
                }
                ```
                """
            )

        # Connect button
        predict_btn.click(
            fn=gradio_predict,
            inputs=[image_input, text_input],
            outputs=[invoice_output, output_image, output_text],
        )

    return demo
