# Releases and production deployment

## Deployment environments

The canonical Vercel project is `invoice` with Root Directory `frontend` and
production branch `main`. Pull requests get Preview deployments. Merging into
`main` creates a Production deployment. Preview and Production are environments,
not Git tags or GitHub Releases. Leave the custom Ignored Build Step empty.

Keep `RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY`, and `RUNPOD_INVOKE_BASE_URL`
server-only in Vercel. Configure inference variables for Preview and Production,
and configure `DEMO_PASSWORD` for Preview only. Vercel Authentication protects
Preview deployments; Production is public and accepts unauthenticated inference
requests. Never commit values or expose them through `NEXT_PUBLIC_`. Use separate
inference credentials and endpoint budgets for Preview and Production.

## Versioning

Use one repository release series: `vMAJOR.MINOR.PATCH`. Add backwards-compatible
features in a minor release and fixes in a patch release. While the project is
pre-1.0, document breaking changes and increment the minor version. Use
`v0.4.0-rc.1` for a release candidate and mark its GitHub Release as a prerelease.
Do not create movable `production`, `preview`, or `latest` Git tags. Keep existing
historical and `data-*` tags intact. Package versions can evolve independently;
the repository release tag identifies the entire source snapshot.

Tag the exact verified `main` commit with an annotated tag, then publish a GitHub
Release with the same name. Never move or reuse a published version tag. Record
the commit, Vercel deployment URL, verification evidence, and relevant backend
image/model versions. Deploy backend images by digest where supported; do not
infer a backend release solely from the frontend version.

## Release checklist

1. Open a PR to `main`; review the diff and resolve actionable review comments.
2. Pass Python checks plus frontend typecheck, build, proxy integration, and
   Chromium review checks. CI uploads repeatable reports and screenshots.
3. Verify the Vercel Preview deployment with a real receipt, both with valid OCR
   and with the OCR field omitted. Check that unauthenticated requests are rejected
   and image-only inference succeeds.
4. Obtain the required approving review and merge without bypassing protection.
5. Verify the new Production deployment uses the merge commit. Confirm the page
   and inference endpoint work without Basic authentication, cross-origin browser
   submissions remain blocked, and a real receipt smoke check succeeds.
6. Only after production passes, create the version tag and GitHub Release.

Configure required status checks on `main` for `test`, `lint`, and `frontend` once
those checks have run. Retain the required approving review. Consider a GitHub tag
ruleset preventing deletion/update of `v*` tags, with narrowly scoped release
creation access.

## Rollback and operational scope

Record the last known-good production deployment before release. If a release
fails, use Vercel's Instant Rollback to restore that deployment, then revert the
faulty change through a PR so `main` agrees with production. Test environment
variable changes too: rollback restores an existing deployment, not a new build
using current secrets. Instant Rollback pauses automatic production-domain
assignment; promote the verified replacement deployment to resume it. Fix forward under a new patch version; retain the old tag
and document the incident in its release notes.

This is a publicly accessible demo: it accepts an image and uses valid
coordinate-bearing OCR when available, falling back to image-only OpenRouter
vision inference otherwise. That fallback sends the invoice image to OpenRouter.
Corrections are saved only in the browser session. Unauthenticated visitors can
submit inference work to Runpod. Before using it as a customer-facing service, add
per-user authentication, durable review/audit storage, distributed request limits,
spending alerts, and a documented retention policy. Monitor failed requests,
inference latency, and Runpod spending; rehearse rollback.

References: [Vercel Git deployments](https://vercel.com/docs/git),
[Vercel Instant Rollback](https://vercel.com/docs/instant-rollback),
[Semantic Versioning](https://semver.org/).
