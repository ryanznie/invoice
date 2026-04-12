# Runpod Deployment

Use this path when you want Triton and the FastAPI backend on one Runpod GPU Pod, with the Next.js UI hosted on Vercel.

## 1. Build and push the Runpod image

From the repo root:

```bash
docker build -f Dockerfile.runpod -t <dockerhub-user>/invoice-ner-runpod:latest .
docker push <dockerhub-user>/invoice-ner-runpod:latest
```

## 2. Create the Runpod Pod

In Runpod:

1. Create a **Secure Cloud** GPU Pod.
2. Choose a GPU that supports your Triton workload.
3. Use **Custom Container Image**:

```text
<dockerhub-user>/invoice-ner-runpod:latest
```

4. Expose HTTP port:

```text
7860
```

5. Optionally expose TCP port `22` if you want full SSH.
6. Attach a **Network Volume** if you want your model data to survive Pod deletion.

Runpod proxy URLs follow this format:

```text
https://<pod-id>-7860.proxy.runpod.net
```

Runpod docs:
- Exposed HTTP ports use `https://[POD_ID]-[INTERNAL_PORT].proxy.runpod.net`
- Network volumes mount to `/workspace` by default on Pods

## 3. Set environment variables on Runpod

Set these Pod env vars:

```env
HOST=0.0.0.0
PORT=7860
LOG_LEVEL=INFO
INFERENCE_BACKEND=triton
TRITON_URL=localhost:8000
TRITON_MODEL_NAME=layoutlmv3-lora-invoice-number
CORS_ORIGINS=http://localhost:3000
CORS_ORIGIN_REGEX=https://.*\.vercel\.app
```

If you move your model repository or use a mounted volume, also set:

```env
TRITON_MODEL_REPOSITORY=/app/triton_model_repo
```

## 4. Verify the backend

Once the Pod is running:

```bash
curl https://<pod-id>-7860.proxy.runpod.net/health
curl https://<pod-id>-7860.proxy.runpod.net/docs
```

## 5. Deploy the frontend to Vercel

In Vercel:

1. Import this repository.
2. Set **Root Directory** to `frontend`.
3. Add this environment variable:

```env
INVOICE_NER_API_URL=https://<pod-id>-7860.proxy.runpod.net
```

4. Deploy.

## 6. Test end to end

Open the Vercel URL and verify:

1. The status card shows `healthy`.
2. Upload an invoice image.
3. Upload the OCR file.
4. Confirm the invoice number and overlay boxes render.
