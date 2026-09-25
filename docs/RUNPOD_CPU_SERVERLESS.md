# Runpod CPU Serverless deployment

This is the production backend used by the Vercel frontend. It is a queue-based
Runpod Serverless worker that scales to zero and accepts an invoice image plus a
coordinate-bearing TXT or JSON OCR file.

## Runtime configuration

| Setting | Value |
| --- | --- |
| Image | `ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.2` |
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
| GHCR image | `ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.2` |
| Image digest | `sha256:e5621052dc83c22def76ae3273b3b3fcfa882d9c1578680d2b68e968fd5bcb1e` |
| Hugging Face model | `ryanznie/layoutlmv3-lora-invoice-number` |
| Hugging Face revision | `104788531d11cba27701c792098403c8f1eb1409` (`onnx-v1`) |
| Public GHCR model mirror | `ghcr.io/ryanznie/layoutlmv3-lora-invoice-number:onnx-v1.0.1` |
| Model mirror digest | `sha256:36d77af14e749233ff0a6f6ff445a6d790e86248bbd3bca21a9f0e74535bf0ba` |
| ONNX SHA-256 | `fff762ae2eb7976f33137fdef64a9cc04c6cbbbe712a3fb0ae39f298fb8386dc` |
| Runpod template | `lfb39n8iuf` |
| Runpod endpoint | `5zp7mr2l2nhbxq` |
| Vercel production URL | `https://frontend-blond-beta-48.vercel.app` |

Both GHCR packages and the Hugging Face model are public. Runpod can pull the
worker without a stored registry credential. The Runpod API key is stored only
as a sensitive Vercel environment variable and in the local Runpod CLI
credential file.

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
  --entrypoint python \
  ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.2 \
  scripts/smoke_test_runpod_backend.py \
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
| runpodctl serverless run 5zp7mr2l2nhbxq --input-file - --wait 15m \
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
  --tag ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.2 \
  --load .
```

Always use an immutable version tag. Do not deploy `latest`.

The application image does not read model weights from the local checkout.
`Dockerfile.runpod.cpu` downloads the model and processor files from the exact
Hugging Face commit shown above, then verifies the ONNX SHA-256 during the
build. The build therefore works from a fresh clone and cannot silently follow
a moving `main` revision.

### Publish a new model bundle

This is a separate, controlled release step and is only required when model or
processor files change. Stage the ONNX artifact, metadata, provenance, model
card, and processor configuration, then upload them in one Hugging Face commit:

```bash
shasum -a 256 models/artifacts/layoutlmv3_invoice_ner.onnx

release_dir=$(mktemp -d)
mkdir -p "$release_dir/onnx"
cp models/artifacts/layoutlmv3_invoice_ner.onnx "$release_dir/onnx/"
cp models/artifacts/model_metadata.json "$release_dir/onnx/"
cp models/artifacts/model_provenance.json "$release_dir/onnx/"
cp models/artifacts/processor_config.json "$release_dir/"
cp models/layoutlmv3-lora-invoice-number/README.md "$release_dir/README.md"

hf upload ryanznie/layoutlmv3-lora-invoice-number "$release_dir" . \
  --commit-message "Publish ONNX production artifact"
```

Read the resulting full Hub commit SHA and pin it, together with the ONNX hash,
in both Runpod Dockerfiles. Never use `main` for a production build. The public
GHCR mirror uses the same canonical model name:

```bash
docker buildx build --platform linux/amd64 \
  --file Dockerfile.runpod.model \
  --tag ghcr.io/ryanznie/layoutlmv3-lora-invoice-number:onnx-v2.0.0 \
  --provenance=false \
  --push .

docker buildx imagetools inspect \
  ghcr.io/ryanznie/layoutlmv3-lora-invoice-number:onnx-v2.0.0
```

Run the PyTorch-to-ONNX parity check and the real invoice container test before
publishing a new worker image. Store the resulting revisions, hashes, and test
results in `model_provenance.json`.

## Push to GHCR

```bash
gh auth token | docker login ghcr.io --username ryanznie --password-stdin
docker push ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.2
```

If the package remains private, create a least-privileged GitHub token with
`read:packages`, register it in Runpod, and pass the returned registry-auth ID to
the template. A public package does not need registry credentials.

## Create the endpoint

Authenticate first with `flash login` or a locally stored `RUNPOD_API_KEY`. Never
commit the key.

For the existing production endpoint, update its template and verify that the
scale-to-zero settings were preserved:

```bash
runpodctl template update lfb39n8iuf \
  --image ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.2
runpodctl serverless get 5zp7mr2l2nhbxq
```

For a new installation, create a serverless template:

```bash
runpodctl template create \
  --name invoice-ner-cpu-v0-3-0 \
  --image ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.2 \
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

The most recent parity, container, and live Runpod results are recorded in
[`artifacts/runpod-onnx-v1-validation.json`](artifacts/runpod-onnx-v1-validation.json).
