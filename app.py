"""
FastAPI + Gradio app for Invoice NER model testing
Accepts: Image + Text file (JSON with words and bboxes)
"""

import json
import torch
from PIL import Image
from typing import List, Dict
from contextlib import asynccontextmanager
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from peft import PeftModel

# Import preprocessing functions from scripts package
from scripts import split_invoice_string, estimate_word_boxes


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "models/layoutlmv3-lora-invoice-number"
BASE_MODEL = "microsoft/layoutlmv3-base"
MAX_LENGTH = 512
NUM_LABELS = 3
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Global model and processor
model = None
processor = None


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_model():
    """Load the LayoutLMv3 model with LoRA adapters"""
    global model, processor

    print(f"🚀 Loading model from {MODEL_PATH}...")
    print(f"📱 Using device: {DEVICE}")

    # Load processor
    processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH, apply_ocr=False)

    # Load base model + LoRA adapter
    base = LayoutLMv3ForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS
    )
    model = PeftModel.from_pretrained(base, MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    print("✅ Model loaded successfully!")


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================


def predict_invoice(
    image: Image.Image, words: List[str], boxes: List[List[int]]
) -> Dict:
    """
    Run inference on invoice

    Args:
        image: PIL Image
        words: List of OCR words
        boxes: List of bounding boxes [x0, y0, x1, y1] normalized to 0-1000

    Returns:
        Dictionary with predictions and invoice number
    """
    if model is None or processor is None:
        raise ValueError("Model not loaded. Call load_model() first.")

    # Validate inputs
    if len(words) != len(boxes):
        raise ValueError(f"Mismatch: {len(words)} words but {len(boxes)} boxes")

    # Process inputs
    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Get word_ids before moving to device
    word_ids = encoding.word_ids(0)

    # Move to device
    encoding_device = {k: v.to(DEVICE) for k, v in encoding.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**encoding_device)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)

        # Get confidence scores
        probs = torch.softmax(logits, dim=2)
        confidence_scores = torch.max(probs, dim=2).values

    # Decode predictions
    predicted_labels = []
    invoice_tokens = []
    word_confidences = []

    token_boxes = encoding["bbox"][0].tolist()

    prev_word_idx = None
    for idx, (pred, box, word_idx, conf) in enumerate(
        zip(
            predictions[0].cpu().tolist(),
            token_boxes,
            word_ids,
            confidence_scores[0].cpu().tolist(),
        )
    ):
        # Skip special tokens and padding
        if box != [0, 0, 0, 0] and word_idx is not None:
            label = model.config.id2label[pred]

            # Only take first subtoken prediction for each word
            if word_idx != prev_word_idx:
                predicted_labels.append(label)
                word_confidences.append(conf)

                # Extract invoice number tokens
                if label.startswith("LABEL_1") or label.startswith("LABEL_2"):
                    invoice_tokens.append(words[word_idx])

                prev_word_idx = word_idx

    # Combine invoice tokens
    invoice_number = " ".join(invoice_tokens) if invoice_tokens else None
    # print(invoice_tokens)
    return {
        "words": words[: len(predicted_labels)],
        "labels": predicted_labels,
        "invoice_number": invoice_number,
        "confidence_scores": word_confidences,
    }


# ============================================================================
# FASTAPI APP
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    load_model()
    yield
    # Cleanup if needed
    print("🔄 Shutting down...")


app = FastAPI(
    title="Invoice NER API",
    description="LayoutLMv3 model for extracting invoice numbers",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """Request model for predictions"""

    words: List[str]
    boxes: List[List[int]]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": DEVICE,
    }


# ============================================================================
# GRADIO INTERFACE
# ============================================================================


def parse_ocr_text_file(file_path: str) -> Dict:
    """
    Parse OCR text file format to JSON
    Uses functions from scripts/preprocess.py for consistency with training data.

    Format: x1,y1,x2,y2,x3,y3,x4,y4,text
    Example: 83,41,331,41,331,78,83,78,TAN WOON YANN

    Steps:
    1. Parse each line as a text box
    2. Split multi-word lines into tokens (split_invoice_string)
    3. Estimate individual word bounding boxes (estimate_word_boxes)

    Args:
        file_path: Path to text file

    Returns:
        Dictionary with words and bboxes
    """
    ocr_entries = []

    # Parse file into entries
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue

            coords = list(map(int, parts[:8]))
            text = ",".join(parts[8:]).strip()

            if not text:
                continue

            # Convert 4-point polygon to bounding box [x0, y0, x1, y1]
            xs, ys = coords[::2], coords[1::2]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            ocr_entries.append({"text": text, "bbox": bbox})

    # Sort by y then x coordinate to ensure reading order
    ocr_entries.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))

    # Split multi-word lines and estimate word boxes
    words = []
    boxes = []

    for entry in ocr_entries:
        text = entry["text"]
        line_bbox = entry["bbox"]

        # Split text into tokens (handles delimiters properly)
        tokens = split_invoice_string(text)

        # Estimate individual word boxes from line box
        token_boxes = estimate_word_boxes(text, tokens, line_bbox)

        # Add words with their estimated boxes
        for token, token_bbox in zip(tokens, token_boxes):
            words.append(token)
            boxes.append(token_bbox)

    return {"words": words, "bboxes": boxes}


def normalize_boxes(
    boxes: List[List[int]], image_width: int, image_height: int
) -> List[List[int]]:
    """
    Normalize bounding boxes to 0-1000 range

    Args:
        boxes: List of boxes in pixel coordinates [x0, y0, x1, y1]
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        List of normalized boxes
    """
    normalized = []
    for box in boxes:
        normalized_box = [
            int(box[0] * 1000 / image_width),
            int(box[1] * 1000 / image_height),
            int(box[2] * 1000 / image_width),
            int(box[3] * 1000 / image_height),
        ]
        normalized.append(normalized_box)
    return normalized


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
        return "❌ Please upload an image", None

    if text_file is None:
        return "❌ Please upload a text file with OCR data", None

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
        else:
            # Parse text file format
            ocr_data = parse_ocr_text_file(file_path)
            words = ocr_data["words"]
            boxes = ocr_data["bboxes"]

            # Normalize boxes to 0-1000 range
            boxes = normalize_boxes(boxes, img_width, img_height)

        if not words or not boxes:
            return (
                "❌ File must contain valid OCR data with words and bounding boxes",
                None,
            )

        # Run prediction
        result = predict_invoice(image, words, boxes)

        # Format output
        output = "## 🎯 Extracted Invoice Number\n\n"
        output += f"### **{result['invoice_number'] or 'Not Found'}**\n\n"
        output += "---\n\n"
        output += "## 📋 Word-level Predictions\n\n"
        output += "| Word | Label | Confidence |\n"
        output += "|------|-------|------------|\n"

        for word, label, conf in zip(
            result["words"], result["labels"], result["confidence_scores"]
        ):
            emoji = (
                "✅"
                if label.startswith("B-INVOICE") or label.startswith("I-INVOICE")
                else "⚪"
            )
            output += f"| {emoji} {word} | `{label}` | {conf:.3f} |\n"

        # Create annotated image
        annotated_image = create_annotated_image(image, words, boxes, result["labels"])

        return output, annotated_image

    except Exception as e:
        return f"❌ Error: {str(e)}", None


def create_annotated_image(image, words, boxes, labels):
    """
    Create an annotated image with bounding boxes

    Args:
        image: PIL Image
        words: List of words
        boxes: List of bounding boxes
        labels: List of predicted labels

    Returns:
        Annotated PIL Image
    """
    from PIL import ImageDraw

    # Create a copy
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Get image dimensions
    width, height = img.size

    for word, box, label in zip(words, boxes, labels):
        # Denormalize coordinates (boxes are in 0-1000 range)
        x0 = int(box[0] * width / 1000)
        y0 = int(box[1] * height / 1000)
        x1 = int(box[2] * width / 1000)
        y1 = int(box[3] * height / 1000)

        # Choose color based on label
        if label.startswith("B-INVOICE") or label.startswith("I-INVOICE"):
            color = "green"
            width_box = 3
        else:
            color = "lightblue"
            width_box = 1

        # Draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width_box)

    return img


# Create Gradio interface
with gr.Blocks(title="Invoice NER Demo") as demo:
    gr.Markdown(
        """
        # 🧾 Invoice Number Extraction
        
        Upload an invoice image and its OCR data to extract the invoice number.
        
        ### 📝 Supported File Formats
        
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

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="📷 Upload Invoice Image", type="pil")
            text_input = gr.File(
                label="📄 Upload OCR Data (TXT or JSON)", file_types=[".txt", ".json"]
            )
            predict_btn = gr.Button("🚀 Extract Invoice Number", variant="primary")

        with gr.Column():
            output_text = gr.Markdown(label="Results")
            output_image = gr.Image(label="📊 Annotated Image")

    # Example
    gr.Markdown("### 💡 How to Use")
    gr.Markdown(
        """
        1. Upload an invoice image (JPG, PNG)
        2. Upload OCR data:
           - **Text file (.txt)**: One line per word in format `x1,y1,x2,y2,x3,y3,x4,y4,text`
           - **JSON file (.json)**: With `words` and `bboxes` fields
        3. Click "Extract Invoice Number"
        
        The model will highlight detected invoice numbers in **green** and other text in **light blue**.
        """
    )

    # Connect button
    predict_btn.click(
        fn=gradio_predict,
        inputs=[image_input, text_input],
        outputs=[output_text, output_image],
    )


# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="Invoice NER App")
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with auto-reload"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run on (default: 7860)"
    )
    args = parser.parse_args()

    # Load model before starting server (unless in debug mode with reload)
    if not args.debug:
        load_model()

    print("\n" + "=" * 60)
    print("🚀 Starting Invoice NER App")
    if args.debug:
        print("🐛 DEBUG MODE - Auto-reload enabled")
    print("=" * 60)
    print(f"📱 Device: {DEVICE}")
    print(f"🌐 Open your browser to: http://localhost:{args.port}")
    print("=" * 60 + "\n")

    uvicorn.run(
        "app:app" if args.debug else app,
        host="0.0.0.0",
        port=args.port,
        reload=args.debug,
        log_level="debug" if args.debug else "info",
    )
