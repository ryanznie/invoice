# Invoice NER Frontend

Next.js frontend for Vercel deployment using `shadcn/ui`.

## Local development

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Set `INVOICE_NER_API_URL` to the FastAPI backend URL. The browser talks to `/api/health` and `/api/predict`; the Next.js server proxies those requests to FastAPI.

## Vercel

1. Import this repository in Vercel.
2. Set the project Root Directory to `frontend`.
3. Add `INVOICE_NER_API_URL` in Vercel environment variables.
4. Deploy.

Recommended setup:

- Frontend project name: `invoice-ner-ui`
- Backend URL value: your stable FastAPI deployment, for example `https://invoice-ner-api.onrender.com`
