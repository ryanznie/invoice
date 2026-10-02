# Browser verification

These integration checks exercise the production UI in Chromium with deterministic
prediction responses. They do not validate the live inference model or Runpod.

Failure cases covered: missing inputs, unsupported or oversized files, unreadable
images, processing errors/non-JSON responses, malformed results, no match, multiple
matches, stale results after changing files, and review status after editing.
Also check keyboard-accessible controls, mobile overflow, copy, reset, and zoom.
Server-side invalid-upload checks hit the real Next.js route without submitting jobs.

Set the same `DEMO_PASSWORD` (at least 32 characters) in both terminal sessions.
Build and start the app, then run with Playwright installed in a separate tools directory:

```sh
npm run build
npm run start -- --port 3100
# In another terminal (npm install --prefix /tmp/receipt-test-tools playwright):
PLAYWRIGHT_MODULE=/tmp/receipt-test-tools/node_modules/playwright \
  node tests/receipt-review.mjs
```

The script uses installed Google Chrome on macOS by default. Set `CHROME_PATH` for
another platform or leave it empty to use Playwright's installed Chromium.
`BASE_URL` defaults to http://localhost:3100. `ARTIFACT_DIR` defaults to
`/tmp/receipt-review-artifacts`. Every run writes a JSON report and screenshots.

## Proxy integration failure matrix

Before changing the proxy, cover these failures through the built Next.js server
and a local HTTP stand-in for Runpod (no paid requests):

- Missing/incorrect demo credentials and absent production password must never
  reach inference; cross-origin browser POSTs must be rejected.
- Malformed multipart, oversized requests, invalid UTF-8/JSON, unequal word/box
  counts, invalid coordinates, and malformed TXT lines must fail before submission.
- Valid JSON (including the `boxes` alias) and TXT must preserve the upload payload.
- Submission rejection, malformed responses, terminal failures, and malformed
  successful output must yield errors, not successful extraction.
- Poll deadline, polling errors, and client disconnection must trigger cancellation
  for a known outstanding job; cancellation failure must be surfaced honestly.
- Health must distinguish ready, busy, initializing, and zero-worker states.

`node tests/proxy-integration.mjs` starts the production server and a local Runpod
stand-in, then saves its report to `/tmp/invoice-proxy-artifacts/report.json`.
Run `npm run build` first. No real credentials are required.

### OCR compatibility and local backend errors

Regression cases: fractional JSON coordinates and TXT files containing skipped
short/empty-text lines must reach inference unchanged. Files with no usable OCR
or invalid numeric rows must still fail before submission. Local backend 4xx
string details must reach the user; structured validation responses need a usable
fallback, and internal 5xx details must not be exposed.
