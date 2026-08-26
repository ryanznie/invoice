# Runpod Backend Deployment

This deployment runs FastAPI and Triton in one Runpod GPU Pod. FastAPI listens on
port `7860`, communicates with Triton on the Pod's private `localhost:8000`, and
is the only service exposed to the Vercel frontend.

## One-time setup

1. Create a Runpod Network Volume in the same datacenter as the Pod.
2. Put the DVC-tracked ONNX file on that volume at:

   ```text
   /workspace/models/artifacts/layoutlmv3_invoice_ner.onnx
   ```

   The repository's tokenizer files and Triton `config.pbtxt` are built into the
   image. Do not commit the ONNX artifact to Git.
3. Add `RUNPOD_API_KEY` as an Actions environment secret named `runpod`.
4. Make the `ghcr.io/<owner>/invoice-ner-runpod` package public, or create a
   Runpod registry credential and supply its ID when running the deploy workflow.

## CI/CD

`Publish Runpod Image` publishes these image tags to GHCR whenever backend
deployment files change on `main`:

- `latest`
- `sha-<commit>`

Use the immutable `sha-<commit>` tag when deploying. From GitHub Actions, run
`Deploy Runpod Pod` with:

- `action=create` to create the initial Pod, with the required Network Volume
  ID and a GPU type such as `NVIDIA RTX A4000` or `NVIDIA RTX A5000`.
- `action=update` and the existing Pod ID to replace its image after a release.

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
