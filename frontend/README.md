# Invoice NER Frontend

Next.js frontend for Vercel deployment using `shadcn/ui`.

## Local development

Start the local API in one terminal from the repository root (ensure the ONNX
model at the configured `MODEL_PATH` is available):

```bash
if [ ! -f .env ]; then cp .env.example .env; fi
uv run uvicorn src.api:app --env-file .env --host 127.0.0.1 --port 7860
```

In another terminal, start the frontend. Copy the example only when you don't
already have a local `.env.local`; it is ignored by Git.

```bash
cd frontend
if [ ! -f .env.local ]; then cp .env.local.example .env.local; fi
npm install
npm run dev
```

The frontend proxy sends requests to `INVOICE_NER_API_URL` when both Runpod
variables are empty. Leave `OPENROUTER_API_KEY` empty in the backend environment
to avoid hosted fallback if local inference fails. The browser talks to
`/api/health` and `/api/predict`; Next.js proxies those requests to FastAPI.
`USE_TORCH=0` skips Transformers' optional PyTorch model import because ONNX
Runtime runs the model directly.

## Vercel

1. Import this repository in Vercel.
2. Set the project Root Directory to `frontend`.
3. Add `RUNPOD_ENDPOINT_ID` and `RUNPOD_API_KEY` in Vercel environment variables. Keep both server-only; never prefix them with `NEXT_PUBLIC_`.
4. Set `DEMO_PASSWORD` to a randomly generated password of at least 32 characters.
5. Deploy. Sign in with username `demo` and that password when the browser prompts.

Recommended setup:

- Canonical Vercel project: `invoice`, Root Directory `frontend`, production branch `main`.
- Leave the custom Ignored Build Step empty so PR previews and production builds run.
- Configure server-only variables for both Preview and Production.
- See [release and rollback workflow](../docs/RELEASES.md).
- The production proxy submits a Runpod Serverless job and polls it to completion, so scale-to-zero cold starts do not expose the Runpod key to the browser.

## Invoice review

The page follows the Gradio demo's image + OCR flow: upload a receipt, extract its
invoice number, compare the result with the original, correct it if needed, and
confirm the number or copy it. The interface centers on a source document beside
extracted data, with explicit review for missing or ambiguous values. Zoom lets users inspect the receipt; expandable
extraction details show the backend's matched words. Red boxes mark the matching
OCR word locations on the source image; the frontend aligns the supplied OCR
coordinates with the backend's word predictions. No other receipt fields are
extracted.

The backend still requires a matching coordinate-bearing TXT/JSON OCR file;
image-only extraction needs an OCR service. Edits and review state are local to
the page and are cleared on refresh or when files change. They are not saved to
the backend.

Uploads are limited to 4 MB combined (OCR: 2 MB maximum) on both client and proxy,
leaving room for multipart headers below [Vercel's 4.5 MB request limit](https://vercel.com/docs/functions/limitations#request-body-size).
Use a smaller JPG/PNG/WebP for larger receipts. The proxy keeps inference secrets
server-side and retains the existing Runpod/local API configuration.

## Verification

Run `npm run lint` and `npm run build`. See [browser verification](tests/README.md)
for the repeatable Chromium checks and screenshot/report artifacts. Prediction
responses in these checks are simulated; verify a real receipt against your
configured inference endpoint after deployment.

## Demo access and job lifecycle

Production builds require `DEMO_PASSWORD` (32+ characters), including when running
`npm run start` locally. Requests fail closed with 503 if it is missing or too short.
Only local development without Runpod credentials may omit it. Page and API access
use HTTP Basic authentication; use HTTPS outside localhost, keep the password
server-only, and share it only with trusted demo participants. This is a private
demo gate, not per-user accounts or a distributed rate limiter. Configure Vercel
Firewall rate limits and Runpod spending/worker limits before broader distribution.
Cross-origin browser submissions are rejected.

OCR is decoded as strict UTF-8 and validated before inference. JSON requires
nonempty string words and one ordered, four-number, finite nonnegative box per word (fractional coordinates are accepted);
`boxes` is accepted as an alias for `bboxes`. Optional `ocr_lines` must be strings.
TXT uses eight integer coordinates followed by text. Short lines and rows with
empty text are skipped, matching the backend parser; at least one usable row is required.
Both declared request length and streamed multipart size are bounded.

The proxy validates successful inference output, bounds upstream calls, and
cancels known outstanding jobs on polling failure, timeout, or client disconnect.
Cancellation uses an independent timeout. An unconfirmed submission/cancellation
asks users to contact the owner before retrying rather than silently resubmitting.
A Runpod job policy bounds its TTL and execution if the proxy is interrupted before
it receives a job ID. `INVOICE_PROCESSING_TIMEOUT_MS` can shorten the default
240-second deadline (minimum 1 second; maximum 240 seconds).

During longer requests, the page checks backend health and reports whether workers
are starting, ready, busy, or idle. The configured Runpod endpoint has a minimum
worker count of zero, so it may scale down between requests; a later job can wait
for startup and model loading. Keeping one worker warm reduces that cold-start wait
but bills for an always-on worker.

Runpod health returns a `ready` boolean and `ready`, `busy`, `initializing`, `idle`,
or `unknown` status instead of declaring every reachable endpoint ready.
