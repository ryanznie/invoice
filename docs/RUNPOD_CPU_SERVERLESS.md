# Runpod CPU Serverless

The queue-based CPU worker accepts a base64-encoded invoice image and TXT or JSON OCR file. The same handler runs locally in tests and in production. OpenRouter is disabled by default.

## Last validated release

The recorded worker image is ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.2. It bundles the ONNX model and processor from Hugging Face revision 104788531d11cba27701c792098403c8f1eb1409; the ONNX SHA-256 is fff762ae2eb7976f33137fdef64a9cc04c6cbbbe712a3fb0ae39f298fb8386dc.

The Dockerfile pins the model revision and verifies the checksum during build. Full image digests, endpoint settings, and validation results are recorded in [runpod-onnx-v1-validation.json](artifacts/runpod-onnx-v1-validation.json).

## Test the backend

Install the development extra first:

    uv sync --extra dev

Run the handler tests without loading model files:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -o addopts='' tests/test_runpod_handler.py tests/test_runpod_smoke_script.py

Run real CPU inference locally. The image and OCR sample must be present in the downloaded SROIE2019 dataset:

    uv run python scripts/smoke_test_runpod_backend.py --image data/SROIE2019/test/img/X00016469670.jpg --ocr data/SROIE2019/test/box/X00016469670.txt --expected PEGIV-1030765

Test the published worker image:

    docker run --rm --platform linux/amd64 --volume "$PWD/data/SROIE2019/test:/fixtures:ro" --entrypoint python ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.2 scripts/smoke_test_runpod_backend.py --image /fixtures/img/X00016469670.jpg --ocr /fixtures/box/X00016469670.txt --expected PEGIV-1030765

For a live endpoint, authenticate runpodctl using its supported credential flow, then generate and submit a payload:

    uv run python scripts/smoke_test_runpod_backend.py --image data/SROIE2019/test/img/X00016469670.jpg --ocr data/SROIE2019/test/box/X00016469670.txt --payload-only | runpodctl serverless run <endpoint-id> --input-file - --wait 15m

This wakes a scale-to-zero worker and may incur usage charges. A timeout only ends the local wait; poll the printed job ID with runpodctl serverless status instead of submitting a duplicate request. Check endpoint health with runpodctl serverless health.

## Build and update

Runpod dependencies are isolated in deploy/runpod/pyproject.toml and pinned by deploy/runpod/uv.lock. After dependency changes:

    uv lock --project deploy/runpod --python 3.11.13
    uv lock --project deploy/runpod --python 3.11.13 --check

Build an immutable Linux AMD64 image from the repository root:
Authenticate Docker to GHCR first; the build command pushes the image.

    docker buildx build --platform linux/amd64 --file deploy/docker/Dockerfile.runpod.cpu --tag ghcr.io/ryanznie/invoice-ner-backend:<version> --push .

The Dockerfile downloads the pinned model and checks its hash, so a local model file is not needed for this build. Publish model or processor changes as a new immutable Hugging Face revision, then update the revision and checksum in both Runpod Dockerfiles and model provenance.

To deploy a worker image, update the template and endpoint:

    runpodctl template update <template-id> --image ghcr.io/ryanznie/invoice-ner-backend:<version>
    runpodctl serverless update <endpoint-id> --template-id <template-id>
    runpodctl serverless get <endpoint-id>

For a new CPU endpoint, create a Serverless template first, then use runpodctl serverless create --template-id <template-id> --compute-type CPU. Keep worker minimum at zero if scale-to-zero behavior is required.

The separately deployed Vercel frontend uses server-side RUNPOD_ENDPOINT_ID, RUNPOD_API_KEY, and RUNPOD_INVOKE_BASE_URL values. Never commit the key.
