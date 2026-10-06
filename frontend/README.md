# Invoice NER Frontend

Next.js frontend for Vercel deployment using `shadcn/ui`.

## Local development

```bash
cd frontend
npm install
# Create frontend/.env.local using the settings below.
npm run dev
```

Create `frontend/.env.local` with the following server-only settings. This local
file is ignored by Git; the setup example lives here instead of a committed
`.env` template. Backend configuration remains in the repository root.

```dotenv
# Local FastAPI backend (used when Runpod credentials are absent)
INVOICE_NER_API_URL=http://localhost:7860

# For local production/Runpod use and Vercel Preview only; username is demo.
# Generate a random password of at least 32 characters.
DEMO_PASSWORD=

# Optional: use Runpod instead of the local backend
RUNPOD_ENDPOINT_ID=
RUNPOD_API_KEY=
RUNPOD_INVOKE_BASE_URL=https://api.runpod.ai/v2

# Optional: processing deadline in milliseconds, clamped to 1000–240000
INVOICE_PROCESSING_TIMEOUT_MS=240000
```

Set `INVOICE_NER_API_URL` to the local FastAPI backend URL. The browser talks to `/api/health` and `/api/predict`; the Next.js server proxies those requests.

## Vercel

1. Import this repository in Vercel.
2. Set the project Root Directory to `frontend`.
3. Add `RUNPOD_ENDPOINT_ID` and `RUNPOD_API_KEY` in Vercel environment variables. Keep both server-only; never prefix them with `NEXT_PUBLIC_`.
4. Set `DEMO_PASSWORD` only for Preview deployments, using a randomly generated password of at least 32 characters.
5. Deploy. Preview requires username `demo` and that password; Production is publicly accessible without an app password.

Recommended setup:

- Canonical Vercel project: `invoice`, Root Directory `frontend`, production branch `main`.
- Leave the custom Ignored Build Step empty so PR previews and production builds run.
- Configure inference variables for both Preview and Production; configure `DEMO_PASSWORD` for Preview only.
- Keep Vercel Authentication enabled for Preview deployments only so the Production URL is public.
- See [release and rollback workflow](../docs/RELEASES.md).
- The production proxy submits a Runpod Serverless job and polls it to completion, so scale-to-zero cold starts do not expose the Runpod key to the browser.

## Invoice review

The page follows the Gradio demo's image + OCR flow: upload a receipt, extract its
invoice number, compare the result with the original, correct it if needed, and
confirm the number or copy it. The interface centers on a source document beside
extracted data, with explicit review for missing or ambiguous values. Zoom lets users inspect the receipt; expandable
extraction details show the backend's matched words. Red boxes mark invoice-number
words on the source image by aligning the supplied OCR coordinates with the
backend predictions and accounts for JPEG EXIF orientation. No other receipt fields are extracted.

The backend uses valid coordinate-bearing TXT/JSON OCR when present. If it is
missing or invalid, the configured OpenRouter vision model extracts directly from
the image. That path has no OCR word coordinates for red-box highlights and sends
the image to OpenRouter, so it requires `OPENROUTER_API_KEY`. Edits and review
state stay in the page and are cleared on refresh or when files change.

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

Vercel Production (`VERCEL_ENV=production`) is publicly accessible and does not
use `DEMO_PASSWORD`. Preview deployments and locally started production builds
require a password of at least 32 characters; Preview and local page/API access use
HTTP Basic authentication. Local development without Vercel or Runpod credentials
may omit the password. The Production bypass uses Vercel's `VERCEL_ENV`, so a
local production build still requires `DEMO_PASSWORD`. Keep the Preview password
server-only. Production still rejects cross-origin browser submissions. Configure
Vercel Firewall rate limits and Runpod spending/worker limits for public access.

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

Runpod health returns a `ready` boolean and `ready`, `busy`, `initializing`, `idle`,
or `unknown` status instead of declaring every reachable endpoint ready.
