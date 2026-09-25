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
| Runpod template | `lfb39n8iuf` |
| Runpod endpoint | `5zp7mr2l2nhbxq` |
| Vercel production URL | `https://frontend-blond-beta-48.vercel.app` |

The GHCR package is public so Runpod can pull it without a stored registry
credential. The Runpod API key is stored only as a sensitive Vercel environment
variable and in the local Runpod CLI credential file.

## Build and validate

Runpod workers are Linux AMD64 even when the image is built from Apple Silicon:

```bash
docker buildx build --platform linux/amd64 \
  --file Dockerfile.runpod.cpu \
  --tag ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.1 \
  --load .
```

Always use an immutable version tag. Do not deploy `latest`.

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
runpodctl serverless run <endpoint-id> --input-file payload.json --wait 10m
```

A scale-to-zero endpoint can show no active workers while idle. The first request
starts a worker and includes image-pull and model-load latency.
