# Invoice NER Frontend

Next.js frontend for Vercel deployment using `shadcn/ui`.

## Local development

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Set `INVOICE_NER_API_URL` to the local FastAPI backend URL. The browser talks to `/api/health` and `/api/predict`; the Next.js server proxies those requests.

## Vercel

1. Import this repository in Vercel.
2. Set the project Root Directory to `frontend`.
3. Add `RUNPOD_ENDPOINT_ID` and `RUNPOD_API_KEY` in Vercel environment variables. Keep both server-only; never prefix them with `NEXT_PUBLIC_`.
4. Deploy.

Recommended setup:

- Frontend project name: `invoice-ner-ui`
- The production proxy submits a Runpod Serverless job and polls it to completion, so scale-to-zero cold starts do not expose the Runpod key to the browser.

## Invoice review

The page follows the Gradio demo's image + OCR flow: upload a receipt, extract its
invoice number, compare the result with the original, correct it if needed, and
confirm the number or copy it. The interface centers on a source document beside
extracted data, with explicit review for missing or ambiguous values. Zoom lets users inspect the receipt; expandable
extraction details show the backend's matched words. No other receipt fields are
extracted. The current API does not return bounding boxes or an annotated image,
so the frontend displays the original image rather than inventing highlights.

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
