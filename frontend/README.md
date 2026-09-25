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
