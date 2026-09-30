# API Contract E2E Failure Matrix

This matrix defines the externally observable failure modes that must be
covered before changing the invoice API contract. The executable check lives
in `tests/e2e/test_api_contract.py` and writes its result to
`docs/artifacts/api-contract-e2e.json`.

| Area | Failure mode | Expected observable result | E2E coverage |
| --- | --- | --- | --- |
| Readiness | Backend or processor is unavailable | `GET /ping` returns `204` | Required |
| Readiness | Backend and processor are available | `GET /ping` returns `200` with `{"status":"healthy"}` | Required |
| Prediction readiness | A request arrives before the runtime is ready | `POST /predict` returns `503` with `Model not loaded` | Required |
| Image limit | Image is exactly `MAX_IMAGE_BYTES` | Request is accepted by the size guard and continues through parsing | Required |
| Image limit | Image is one byte over `MAX_IMAGE_BYTES` | `POST /predict` returns `413` before image decoding | Required |
| OCR limit | OCR is exactly `MAX_OCR_BYTES` | Request is accepted by the size guard and continues through parsing | Required |
| OCR limit | OCR is one byte over `MAX_OCR_BYTES` | `POST /predict` returns `413` before OCR parsing | Required |
| Image parsing | Uploaded bytes are not an image | `POST /predict` returns `400` | Required |
| OCR parsing | JSON is malformed, content is empty, or the extension is unsupported | `POST /predict` returns `400` with a stable public error | Required |
| Processor bundle | The image contains a local processor bundle next to the ONNX model | Startup loads that local bundle without a Hugging Face network dependency | Required in the offline production-container run |
| Processor fallback | The configured processor path is unusable | Startup attempts `BASE_MODEL`; startup fails if neither source is available | Required in the offline production-container run |
| Missing OpenRouter key | Primary inference fails and `OPENROUTER_API_KEY` is unavailable | `POST /predict` returns `503` with a clear key requirement and makes no hosted request | Required |
| Invalid OpenRouter settings | Hosted fallback settings contain nonnumeric values | Model startup succeeds; a request needing fallback returns `503` with a clear configuration error and makes no hosted request | Required |
| Inference fallback succeeds | Primary inference fails and OpenRouter succeeds | `POST /predict` returns the fallback invoice result | Required with a deterministic local double; no external request is made |
| Inference fallback failure | Primary inference and OpenRouter both fail | `POST /predict` returns the sanitized `500` response | Required with a deterministic local double; no external request is made |

The processor cases intentionally use `tests/e2e/test_processor_startup.py`
against the production container because only that image contains the real
ONNX model and complete processor bundle. The request-contract cases use the
real ASGI routes and multipart parsing with deterministic local runtime doubles
so they do not require model downloads, credentials, or billable inference.
