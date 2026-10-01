# Browser verification

These integration checks exercise the production UI in Chromium with deterministic
prediction responses. They do not validate the live inference model or Runpod.

Failure cases covered: missing inputs, unsupported or oversized files, unreadable
images, processing errors/non-JSON responses, malformed results, no match, multiple
matches, stale results after changing files, and review status after editing.
Also check keyboard-accessible controls, mobile overflow, copy, reset, and zoom.
Server-side invalid-upload checks hit the real Next.js route without submitting jobs.

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
