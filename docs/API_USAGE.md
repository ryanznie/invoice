# API Usage Guide

Complete guide for using the Invoice NER API endpoints.

## 📋 Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Predict Invoice Number](#predict-invoice-number)
- [Data Formats](#data-formats)
- [Error Handling](#error-handling)
- [Code Examples](#code-examples)
- [Rate Limits](#rate-limits)

## Overview

The Invoice NER API provides two main endpoints:
1. **Health Check** - Check if the API and model are ready
2. **Predict** - Extract invoice numbers from invoice images with OCR data

The API uses a **two-stage extraction pipeline**:
1. **Heuristic Extraction** - Fast pattern matching for common invoice number formats
2. **Model Inference** - LayoutLMv3-based deep learning when heuristics fail

## Base URL

```
http://localhost:7860
```

Replace with your actual deployment URL.

## Endpoints

### Health Check

Check if the API is running and the model is loaded.

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "mps"
}
```

**Fields:**
- `status` (string): `"healthy"` or `"unhealthy"`
- `model_loaded` (boolean): Whether the model is loaded and ready
- `device` (string): Device the model is running on (`cpu`, `cuda`, or `mps`)

**Example:**

```bash
curl http://localhost:7860/health
```

---

### Predict Invoice Number

Extract invoice number from an invoice image and OCR data.

**Endpoint:** `POST /predict`

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | File | Yes | Invoice image (JPG, PNG, etc.) |
| `ocr_file` | File | Yes | OCR data in TXT or JSON format |

**OCR File Formats:**

#### 1. Text Format (.txt)

Each line contains: `x1,y1,x2,y2,x3,y3,x4,y4,text`

```
83,41,331,41,331,78,83,78,TAN WOON YANN
352,37,542,37,542,72,352,72,BOOK TA .K(TAMAN DAYA) SDN BHD
83,82,626,82,626,116,83,116,Registration No. : 002177368-W
...
```

Where:
- `x1,y1,x2,y2,x3,y3,x4,y4`: Bounding box coordinates (quadrilateral)
- `text`: OCR recognized text

#### 2. JSON Format (.json)

```json
{
  "words": ["TAN", "WOON", "YANN", "BOOK", ...],
  "bboxes": [
    [83, 41, 331, 78],
    [352, 37, 542, 72],
    ...
  ],
  "ocr_lines": [
    "TAN WOON YANN",
    "BOOK TA .K(TAMAN DAYA) SDN BHD",
    ...
  ]
}
```

**Fields:**
- `words` (array of strings): Individual words from OCR
- `bboxes` or `boxes` (array of arrays): Bounding boxes `[x0, y0, x1, y1]`
- `ocr_lines` (array of strings, optional): Full text lines for heuristic matching

**Coordinate Systems:**

The API accepts both:
1. **Normalized coordinates** (0-1000 range) - preferred
2. **Pixel coordinates** - automatically normalized based on image dimensions

**Response:**

```json
{
  "invoice_number": "INV-2023-001234",
  "extraction_method": "heuristic",
  "predictions": [
    {
      "word": "Invoice",
      "label": "LABEL_0",
      "confidence": 0.9823,
      "is_invoice_number": false
    },
    {
      "word": "INV-2023-001234",
      "label": "HEURISTIC_MATCH",
      "confidence": 1.0,
      "is_invoice_number": true
    },
    ...
  ],
  "total_words": 127,
  "model_device": "mps"
}
```

**Response Fields:**

- `invoice_number` (string): Extracted invoice number or `"Not Found"`
- `extraction_method` (string): `"heuristic"` or `"model"`
- `predictions` (array): Word-level predictions
  - `word` (string): The word/token
  - `label` (string): Predicted label (`LABEL_0`, `LABEL_1`, `LABEL_2`, or `HEURISTIC_MATCH`)
  - `confidence` (float): Model confidence score (0-1)
  - `is_invoice_number` (boolean): Whether this word is part of the invoice number
- `total_words` (integer): Total number of words processed
- `model_device` (string): Device used for inference

**Label Meanings:**

- `LABEL_0`: Not an invoice number (O - Outside)
- `LABEL_1`: Beginning of invoice number (B-INVOICE_NUMBER)
- `LABEL_2`: Inside/continuation of invoice number (I-INVOICE_NUMBER)
- `HEURISTIC_MATCH`: Matched by heuristic rules

**Example:**

```bash
curl -X POST http://localhost:7860/predict \
  -F "image=@invoice.jpg" \
  -F "ocr_file=@ocr_data.json"
```

**Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid file format, missing data) |
| 503 | Service unavailable (model not loaded) |
| 500 | Internal server error |

---

## Data Formats

### Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)
- TIFF (`.tiff`, `.tif`)
- WebP (`.webp`)

All images are converted to RGB internally.

### OCR Data Requirements

**Text Format (.txt):**
- One bounding box per line
- 8 coordinates + comma + text
- Coordinates are quadrilateral corners (clockwise from top-left)

**JSON Format (.json):**
- `words`: Non-empty array of strings
- `bboxes`/`boxes`: Array of 4-element arrays `[x0, y0, x1, y1]`
- `ocr_lines`: Optional array of strings (recommended for better heuristics)

**Validation:**
- `len(words)` must equal `len(boxes)`
- Each box must have 4 numeric coordinates
- Coordinates must form valid rectangles (x0 < x1, y0 < y1)

---

## Error Handling

### Common Errors

#### 400 Bad Request

```json
{
  "detail": "OCR file must be .txt or .json format"
}
```

**Causes:**
- Wrong file extension
- Invalid JSON syntax
- Missing required fields
- Mismatched words and boxes lengths

#### 503 Service Unavailable

```json
{
  "detail": "Model not loaded"
}
```

**Solution:** Wait for model to load, check `/health` endpoint

#### 500 Internal Server Error

```json
{
  "detail": "Internal server error: ..."
}
```

**Causes:**
- Corrupted image file
- Out of memory
- Model inference error

---

## Code Examples

### Python with `requests`

```python
import requests

# Health check
response = requests.get("http://localhost:7860/health")
print(response.json())

# Predict invoice number
with open("invoice.jpg", "rb") as img, open("ocr_data.json", "rb") as ocr:
    files = {
        "image": ("invoice.jpg", img, "image/jpeg"),
        "ocr_file": ("ocr_data.json", ocr, "application/json")
    }
    response = requests.post("http://localhost:7860/predict", files=files)
    result = response.json()
    
    print(f"Invoice Number: {result['invoice_number']}")
    print(f"Method: {result['extraction_method']}")
    print(f"Confidence: {result['predictions'][0]['confidence']}")
```

### Python with `httpx` (Async)

```python
import httpx
import asyncio

async def extract_invoice():
    async with httpx.AsyncClient() as client:
        # Health check
        health = await client.get("http://localhost:7860/health")
        print(health.json())
        
        # Predict
        with open("invoice.jpg", "rb") as img, open("ocr_data.json", "rb") as ocr:
            files = {
                "image": ("invoice.jpg", img),
                "ocr_file": ("ocr_data.json", ocr)
            }
            response = await client.post(
                "http://localhost:7860/predict",
                files=files,
                timeout=30.0
            )
            return response.json()

result = asyncio.run(extract_invoice())
print(result["invoice_number"])
```

### cURL

```bash
# Health check
curl http://localhost:7860/health

# Predict with JSON OCR data
curl -X POST http://localhost:7860/predict \
  -F "image=@invoice.jpg" \
  -F "ocr_file=@ocr_data.json" \
  | jq .

# Predict with TXT OCR data
curl -X POST http://localhost:7860/predict \
  -F "image=@invoice.png" \
  -F "ocr_file=@ocr_data.txt" \
  | jq '.invoice_number'
```

### JavaScript (Node.js)

```javascript
const fs = require('fs');
const FormData = require('form-data');
const axios = require('axios');

async function extractInvoice() {
  const form = new FormData();
  form.append('image', fs.createReadStream('invoice.jpg'));
  form.append('ocr_file', fs.createReadStream('ocr_data.json'));
  
  const response = await axios.post('http://localhost:7860/predict', form, {
    headers: form.getHeaders(),
    timeout: 30000
  });
  
  console.log('Invoice Number:', response.data.invoice_number);
  console.log('Method:', response.data.extraction_method);
  return response.data;
}

extractInvoice().catch(console.error);
```

### JavaScript (Browser)

```javascript
async function uploadInvoice(imageFile, ocrFile) {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('ocr_file', ocrFile);
  
  const response = await fetch('http://localhost:7860/predict', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }
  
  const result = await response.json();
  console.log('Invoice Number:', result.invoice_number);
  return result;
}

// Usage with file input
document.getElementById('submitBtn').addEventListener('click', async () => {
  const imageFile = document.getElementById('imageInput').files[0];
  const ocrFile = document.getElementById('ocrInput').files[0];
  
  try {
    const result = await uploadInvoice(imageFile, ocrFile);
    document.getElementById('result').textContent = result.invoice_number;
  } catch (error) {
    console.error('Error:', error);
  }
});
```

---

## Rate Limits

**Current version:** No rate limits enforced.

For production, consider implementing:
- Request rate limiting (e.g., 100 requests/minute per IP)
- Concurrent request limits
- File size limits (currently unlimited)

**Recommended Limits:**
- Max image size: 10MB
- Max OCR file size: 1MB
- Timeout: 30 seconds per request

---

## Interactive Documentation

FastAPI provides interactive API documentation:

- **Swagger UI**: http://localhost:7860/docs
- **ReDoc**: http://localhost:7860/redoc

These interfaces allow you to:
- View all endpoints and schemas
- Try API calls directly from the browser
- Download OpenAPI specification

---

## Best Practices

### 1. Always Check Health First

```python
response = requests.get(f"{base_url}/health")
if not response.json().get("model_loaded"):
    raise Exception("Model not ready")
```

### 2. Handle Errors Gracefully

```python
try:
    response = requests.post(url, files=files, timeout=30)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except Exception as e:
    print(f"Error: {e}")
```

### 3. Use Appropriate Timeouts

```python
# Model inference can take 5-15 seconds
response = requests.post(url, files=files, timeout=30)
```

### 4. Validate Input Data

```python
# Check file sizes
if os.path.getsize(image_path) > 10 * 1024 * 1024:  # 10MB
    raise ValueError("Image too large")

# Validate JSON structure
with open(ocr_path) as f:
    data = json.load(f)
    if "words" not in data or "bboxes" not in data:
        raise ValueError("Invalid OCR JSON format")
```

### 5. Process Results

```python
result = response.json()

# Extract invoice number
invoice_num = result["invoice_number"]
if invoice_num == "Not Found":
    print("No invoice number detected")
else:
    print(f"Found: {invoice_num}")

# Get high-confidence predictions
confident_words = [
    p for p in result["predictions"]
    if p["is_invoice_number"] and p["confidence"] > 0.9
]
```

---

## Gradio UI Alternative

In addition to the REST API, you can use the **Gradio web interface** at:

```
http://localhost:7860/
```

Features:
- Drag-and-drop file upload
- Real-time preview
- Visual bounding boxes
- No coding required

Perfect for:
- Manual testing
- Demos
- Non-technical users

---

## Troubleshooting

### Model Not Loading

**Check logs:**
```bash
docker-compose logs -f
```

**Verify model files:**
```bash
ls -lh models/layoutlmv3-lora-invoice-number/
```

### Slow Inference

**Check device:**
```python
response = requests.get("http://localhost:7860/health")
print(response.json()["device"])  # Should be 'cuda' or 'mps' for GPU
```

**CPU inference takes 5-15 seconds per image**
**GPU (CUDA/MPS) takes 1-3 seconds per image**

### Memory Issues

**Reduce Docker memory limits:**
```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
```

**Or set in `.env`:**
```bash
DOCKER_MEMORY_LIMIT=6G
```

---

## Support

- **Documentation**: [README.md](../README.md)
- **Testing Guide**: [TESTING.md](TESTING.md)
- **Development Setup**: [DEV_SETUP.md](DEV_SETUP.md)
- **Issues**: Open an issue on GitHub

---

## Version History

- **v1.0.0** (2025-01-10)
  - Initial API release
  - Health check endpoint
  - Predict endpoint with multipart file upload
  - Support for TXT and JSON OCR formats
  - Heuristic + Model extraction pipeline
