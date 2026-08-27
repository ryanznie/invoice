# Runpod Backend Deployment

The default deployment runs FastAPI and Triton in one Runpod GPU Pod. FastAPI
listens on port `7860`, communicates with Triton on the Pod's private
`localhost:8000`, and is the only service exposed to the Vercel frontend.

For temporary low-cost deployments, the workflow can also create a CPU Pod that
runs FastAPI directly with ONNX Runtime. CPU inference is slower and lower
throughput than the GPU/Triton path, but it is useful when GPUs are unavailable
or the app only needs light demo traffic.

## One-time setup

1. Create a Runpod Network Volume in the same datacenter where the Pod will run.
   The volume is persistent storage for the model file. Without it, the Pod can
   start the image but cannot load the ONNX model.
2. Put the DVC-tracked ONNX file on that volume at:

   ```text
   /workspace/models/artifacts/layoutlmv3_invoice_ner.onnx
   ```

   The repository's tokenizer files and Triton `config.pbtxt` are built into the
   image. Do not commit the ONNX artifact to Git.
3. Add `RUNPOD_API_KEY` as an Actions environment secret in the GitHub
   environment named `runpod`. This lets the deploy workflow create or update
   the Pod through the Runpod API without exposing the key in the repository.
4. Make the GHCR package public, or create a Runpod registry credential and
   supply its ID when running the deploy workflow. The Pod must be able to pull
   the GHCR image; public GHCR is simplest for the first deployment.

## CI/CD

`Publish Runpod Image` publishes these image tags to GHCR whenever backend
deployment files change on `main`:

- `latest`
- `sha-<commit-sha>`

It publishes two images with the same tag:

- `ghcr.io/<owner>/invoice-ner-runpod` for the default GPU/Triton backend.
- `ghcr.io/<owner>/invoice-ner-runpod-cpu` for the CPU/ONNX backend.

Use the immutable `sha-<commit-sha>` tag when deploying. In the completed
`Publish Runpod Image` workflow run, open the `Compute image tags` or
`Build and publish GPU image` step and copy the tag that looks like:

```text
ghcr.io/<owner>/invoice-ner-runpod:sha-abc1234
```

When the deploy workflow asks for `image_tag`, enter only the tag portion:

```text
sha-abc1234
```

From GitHub Actions, run `Deploy Runpod Pod` with:

- `action=create` to create the initial Pod, with the required Network Volume
  ID and a GPU type such as `NVIDIA RTX A4000` or `NVIDIA RTX A5000`.
- `action=update` and the existing Pod ID to replace its image after a release.

For the default production path, use:

```text
backend_target=gpu-triton
gpu_type=NVIDIA RTX A4000
gpu_count=1
vcpu_count=<leave default>
```

For a temporary CPU deployment, use:

```text
backend_target=cpu-onnx
gpu_type=<ignored>
gpu_count=<ignored>
vcpu_count=4
```

The workflow configures the Pod with `7860/http`; Runpod assigns the backend URL:

```text
https://<pod-id>-7860.proxy.runpod.net
```

## Connect Vercel

Set the Vercel production environment variable and redeploy the frontend:

```bash
vercel env update INVOICE_NER_API_URL production \
  https://<pod-id>-7860.proxy.runpod.net
vercel deploy --prod
```

Verify the backend before deploying the frontend:

```bash
curl --fail https://<pod-id>-7860.proxy.runpod.net/health
```

The Vercel app proxies browser uploads through its `/api/predict` route, so the
backend CORS policy is a defense-in-depth setting rather than a requirement for
the normal production request path.
