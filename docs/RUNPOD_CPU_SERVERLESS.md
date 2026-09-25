# Runpod CPU Serverless deployment

This is the production backend used by the Vercel frontend. It is a queue-based
Runpod Serverless worker that scales to zero and accepts an invoice image plus a
coordinate-bearing TXT or JSON OCR file.

## Runtime configuration

| Setting | Value |
| --- | --- |
| Image | `ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.1` |
| Compute | CPU, `cpu5c-4-8` (4 vCPU / 8 GB) |
| Workers | minimum 0, maximum 2 |
| Idle timeout | 300 seconds |
| Execution timeout | 300 seconds |
| Inference | heuristics, then FP32 LayoutLMv3 ONNX fallback |
| Model | `/app/models/artifacts/layoutlmv3_invoice_ner.onnx` |
| OpenRouter | disabled by default |

The model and processor files are baked into the image. A network volume is not
required.

## Production resources

| Resource | Value |
| --- | --- |
| GHCR image | `ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.1` |
| Image digest | `sha256:f445cb938f383948ecf3206b8265438c17652ac260b841933daf4ea55a288e23` |
| Model bundle | `ghcr.io/ryanznie/invoice-ner-backend:model-layoutlmv3-onnx-v1` |
| Model bundle digest | `sha256:4af70102cccce71e065ad9568b085074177ef5021ce7f8f849a0eab28fa2be4e` |
| ONNX SHA-256 | `fff762ae2eb7976f33137fdef64a9cc04c6cbbbe712a3fb0ae39f298fb8386dc` |
| Runpod template | `lfb39n8iuf` |
| Runpod endpoint | `5zp7mr2l2nhbxq` |
| Vercel production URL | `https://frontend-blond-beta-48.vercel.app` |

The GHCR package is public so Runpod can pull it without a stored registry
credential. The Runpod API key is stored only as a sensitive Vercel environment
variable and in the local Runpod CLI credential file.

## Backend-only testing

These checks bypass Vercel completely. The test fixture below includes both the
invoice image and its bounding-box OCR data.

### 1. Run the fast handler tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -o addopts='' \
  tests/test_runpod_handler.py \
  tests/test_runpod_smoke_script.py
```

This validates request decoding, file validation, and payload generation without
loading the model or starting a Runpod worker.

### 2. Run the real backend locally

```bash
uv run python scripts/smoke_test_runpod_backend.py \
  --image data/SROIE2019/test/img/X00016469670.jpg \
  --ocr data/SROIE2019/test/box/X00016469670.txt \
  --expected PEGIV-1030765
```

This calls the same `runpod_handler.handler` function used in production and
runs the real CPU inference stack in the current Python environment. A mismatch
returns a non-zero exit code, which makes the command suitable for CI.

### 3. Test the production container locally

Build the image as described in [Build and validate](#build-and-validate), then
run its handler with the fixture directory mounted read-only:

```bash
docker run --rm --platform linux/amd64 \
  --volume "$PWD/data/SROIE2019/test:/fixtures:ro" \
  --volume "$PWD/scripts:/app/test-scripts:ro" \
  --entrypoint python \
  ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.1 \
  test-scripts/smoke_test_runpod_backend.py \
  --image /fixtures/img/X00016469670.jpg \
  --ocr /fixtures/box/X00016469670.txt \
  --expected PEGIV-1030765
```

This catches missing dependencies, model artifacts, and image architecture
problems before a deployment.

### 4. Call the live Runpod endpoint directly

Install or update `runpodctl`, then authenticate once with `runpodctl doctor` or
set `RUNPOD_API_KEY` in your shell. Never put the key in the command, payload, or
repository.

Check the endpoint first:

```bash
runpodctl serverless health 5zp7mr2l2nhbxq
```

Generate the base64 payload and send it directly to Runpod:

```bash
uv run python scripts/smoke_test_runpod_backend.py \
  --image data/SROIE2019/test/img/X00016469670.jpg \
  --ocr data/SROIE2019/test/box/X00016469670.txt \
  --payload-only \
| runpodctl serverless run 5zp7mr2l2nhbxq --input - --wait 15m \
| jq '{status, delayTime, executionTime, output: {
    invoice_number: .output.invoice_number,
    extraction_method: .output.extraction_method,
    model_device: .output.model_device
  }}'
```

`runpodctl` adds the outer `{"input": ...}` envelope, so the generated JSON
must not include it. The expected invoice number is `PEGIV-1030765`. Because the
endpoint scales to zero, the first request after an idle period also includes
container startup and model-load time. It wakes a billable worker for the job;
the worker becomes eligible to scale back to zero after the 300-second idle
timeout.

If a job exceeds the local wait budget, do not submit it again. Use the job ID
printed by `runpodctl`:

```bash
runpodctl serverless status 5zp7mr2l2nhbxq <job-id> --wait 10m
```

For a job that is stuck or failed, inspect health and recent worker logs:

```bash
runpodctl serverless health 5zp7mr2l2nhbxq
runpodctl serverless logs 5zp7mr2l2nhbxq --since 15m
```

## Build and validate

Runpod workers are Linux AMD64 even when the image is built from Apple Silicon:

```bash
docker buildx build --platform linux/amd64 \
  --file Dockerfile.runpod.cpu \
  --tag ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.1 \
  --load .
```

Always use an immutable version tag. Do not deploy `latest`.

The application image does not read model weights from the local checkout.
`Dockerfile.runpod.cpu` copies the model and processor files from the public
model-bundle image pinned by its immutable manifest digest, then verifies the
ONNX file SHA-256 during the build. This means the command above works from a
fresh clone containing only the tracked DVC pointer.

### Publish a new model bundle

This is a separate, controlled release step and is only required when model or
processor files change. Retrieve the DVC artifact on a machine authorized to
read the model remote, verify the expected file, then publish a new immutable
model tag:

```bash
dvc pull models/artifacts/layoutlmv3_invoice_ner.onnx.dvc
shasum -a 256 models/artifacts/layoutlmv3_invoice_ner.onnx

docker buildx build --platform linux/amd64 \
  --file Dockerfile.runpod.model \
  --tag ghcr.io/ryanznie/invoice-ner-backend:model-layoutlmv3-onnx-v2 \
  --provenance=false \
  --push .

docker buildx imagetools inspect \
  ghcr.io/ryanznie/invoice-ner-backend:model-layoutlmv3-onnx-v2
```

Update both `MODEL_BUNDLE_IMAGE` and `MODEL_SHA256` in
`Dockerfile.runpod.cpu` to the newly published manifest and file digests. Never
reference a model bundle by tag alone in the backend Dockerfile.

## Push to GHCR

```bash
gh auth token | docker login ghcr.io --username ryanznie --password-stdin
docker push ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.1
```

If the package remains private, create a least-privileged GitHub token with
`read:packages`, register it in Runpod, and pass the returned registry-auth ID to
the template. A public package does not need registry credentials.

## Create the endpoint

Authenticate first with `flash login` or a locally stored `RUNPOD_API_KEY`. Never
commit the key.

```bash
runpodctl template create \
  --name invoice-ner-cpu-v0-3-0 \
  --image ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.1 \
  --serverless \
  --container-disk-in-gb 5
```

Use the returned template ID:

```bash
runpodctl serverless create \
  --name invoice-ner-production \
  --template-id <template-id> \
  --compute-type CPU \
  --instance-id cpu5c-4-8 \
  --workers-min 0 \
  --workers-max 2 \
  --idle-timeout 300 \
  --scale-by requests \
  --scale-threshold 1
```

Do not use the generic REST endpoint to create CPU workers; use `runpodctl` with
`--compute-type CPU`.

## Vercel variables

Set these as server-side Vercel environment variables. Do not prefix them with
`NEXT_PUBLIC_`.

```env
RUNPOD_ENDPOINT_ID=<endpoint-id>
RUNPOD_API_KEY=<secret>
RUNPOD_INVOKE_BASE_URL=https://api.runpod.ai/v2
```

Set the Vercel project root directory to `frontend`.

## Verification

The frontend submits an asynchronous `/run` job and polls its status. For direct
verification:

```bash
runpodctl serverless health <endpoint-id>
runpodctl serverless run <endpoint-id> --input-file payload.json --wait 15m
```

A scale-to-zero endpoint can show no active workers while idle. The first request
starts a worker and includes image-pull and model-load latency. See
[Backend-only testing](#backend-only-testing) for generating a real payload and
testing locally, inside Docker, or against the deployed endpoint.
