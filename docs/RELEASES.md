# Releases and production deployment

## Deployment environments

The canonical Vercel project is `invoice` with Root Directory `frontend` and
production branch `main`. Pull requests get Preview deployments. Merging into
`main` creates a Production deployment. Preview and Production are environments,
not Git tags or GitHub Releases. Leave the custom Ignored Build Step empty.

Keep `RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY`, `RUNPOD_INVOKE_BASE_URL`, and
`DEMO_PASSWORD` server-only in Vercel. Configure Preview and Production explicitly;
never commit values or expose them through `NEXT_PUBLIC_`. For this trusted demo,
both environments may use the same inference endpoint. Use separate credentials,
passwords, and endpoint budgets before expanding access.

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
3. Verify the Vercel Preview deployment with a real receipt and its OCR file.
   Check that unauthenticated requests are rejected and inference succeeds.
4. Obtain the required approving review and merge without bypassing protection.
5. Verify the new Production deployment uses the merge commit, then repeat the
   authentication and real receipt smoke checks on the production URL.
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

This is a password-protected demo: it requires an image plus coordinate-bearing
OCR and saves corrections only in the browser session. Before offering public
customer access, add per-user authentication, durable review/audit storage,
distributed request limits, spending alerts, and a documented retention policy.
Monitor failed requests, inference latency, and Runpod spending; rehearse rollback.

References: [Vercel Git deployments](https://vercel.com/docs/git),
[Vercel Instant Rollback](https://vercel.com/docs/instant-rollback),
[Semantic Versioning](https://semver.org/).
