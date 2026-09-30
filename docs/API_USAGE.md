# API

The local app serves the FastAPI API and Gradio UI on port 7860. Open /docs for the interactive API schema.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /health | Model and backend status |
| GET | /ping | Runpod readiness: 200 when ready, 204 while loading or unavailable |
| GET | /metrics | Prometheus metrics |
| GET | /runtime/config | Non-secret inference configuration |
| POST | /predict | Extract an invoice number |

There is no built-in API authentication or request-rate limit. Put public deployments behind an authenticated gateway or reverse proxy.

## Predict

Send a multipart request with an image and an OCR file:

    curl -F 'image=@invoice.jpg' -F 'ocr_file=@ocr.json' http://localhost:7860/predict

The image must be readable by Pillow. The OCR filename must end in .txt or .json.

TXT OCR files contain one line per OCR region, with eight polygon coordinates followed by text:

    x1,y1,x2,y2,x3,y3,x4,y4,text

JSON OCR files use words and bounding boxes:

    {
      "words": ["Invoice", "No.", "INV-12345"],
      "bboxes": [[10, 20, 80, 45], [82, 20, 110, 45], [120, 20, 250, 45]],
      "ocr_lines": ["Invoice No. INV-12345"]
    }

Use one [x0, y0, x1, y1] box per word. TXT coordinates are normalized using the image dimensions. JSON boxes should use the 0–1000 range; JSON coordinates above 1000 are treated as pixels and normalized automatically.

A successful response includes invoice_number, extraction_method, predictions, total_words, and model_device. extraction_method is heuristic or model. If no match is found, invoice_number is Not Found.

## Limits and errors

Default upload limits are 10 MiB for images and 2 MiB for OCR files; configure them with MAX_IMAGE_BYTES and MAX_OCR_BYTES.

| Status | Meaning |
| --- | --- |
| 400 | Invalid image or OCR file |
| 413 | Upload exceeds its configured size limit |
| 422 | Required multipart fields are missing |
| 500 | Inference failed |
| 503 | Model is not ready, or hosted fallback configuration is missing or invalid |

Internal error details are hidden by default. Do not enable EXPOSE_INTERNAL_ERRORS in production.

## Hosted fallback

When local model inference raises an error, the app sends the image and OCR words to the configured OpenRouter model. Set `OPENROUTER_API_KEY` only when hosted processing is approved; without it, the request returns a clear `503` error.

**Migration:** `ENABLE_OPENROUTER_FALLBACK` is deprecated and ignored. The app logs a warning if it is still set. Deployments that must keep invoice data local should remove `OPENROUTER_API_KEY`; local processing continues, and a request that needs hosted fallback returns `503`.
