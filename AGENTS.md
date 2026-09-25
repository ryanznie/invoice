# AGENTS.md

## Git
- Create branches with semantic prefixes: `feature/`, `fix/`, `chore/`, `docs/`, `refactor/`, or `test/`.
- Do not use `codex/` branch names in this repository.
- Do not revert or overwrite uncommitted user changes unless explicitly asked.
- Prefer focused commits using Conventional Commit messages.

## Validation
- Run relevant tests before finishing.
- For backend changes, prefer `uv run pytest`.
- For frontend changes, run the project's lint and build checks when available.
- Report any tests that could not be run.

## Code Style
- Follow existing project patterns.
- Keep changes scoped to the request.
- Avoid unrelated refactors.
- Add dependencies only when they clearly reduce complexity or match the existing stack.

## Repo Hygiene
- Do not commit secrets, `.env*` files, generated datasets, model artifacts, `wandb/`, or local cache files.
- Treat `data/`, `models/`, `wandb/`, `.vercel/`, and similar generated output directories as user-owned unless the task explicitly targets them.
- Update docs when setup, API behavior, deployment, or developer workflow changes.
