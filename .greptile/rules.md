# Code Review Rules

## Correctness

- Prioritize issues that can change extracted invoice numbers, confidence handling, fallback behavior, or API responses.
- Check edge cases around empty OCR, malformed OCR, missing images, ambiguous labels, and model loading failures.
- Flag changes that make heuristic extraction and model inference disagree without clear resolution logic.

## API and Frontend Contract

- Backend response fields used by the frontend should remain stable unless docs, tests, and UI handling are updated together.
- Frontend upload and proxy routes should handle API errors without hiding useful diagnostics from users.

## Security and Privacy

- Treat invoice images, OCR text, labels, and model artifacts as sensitive project data.
- Do not commit secrets, `.env*` files, generated datasets, model weights, W&B logs, local caches, or large binary artifacts.
- Validate uploaded filenames, content types, payload sizes, and parsed OCR before passing data to inference code.

## ML and Deployment

- Model path, device, timeout, and fallback changes should be reviewed against Docker, RunPod, Triton, and docs configuration.
- Training or benchmarking changes should avoid data leakage between train, validation, and test sets.
- Prefer deterministic, testable postprocessing over broad prompt or heuristic changes without examples.

## PR Understanding Quiz

- Include a "PR Understanding Quiz" section in each PR summary.
- Write 3-5 multiple-choice questions that help the author check their understanding of the changed code paths, contracts, tradeoffs, and risks.
- Each question should have exactly four answer options labeled A-D.
- Keep questions grounded in the PR diff and repository context; avoid generic coding trivia.
- Include a readable answer key immediately after the quiz in a collapsible Markdown details block:

```markdown
<details>
<summary>Answer key</summary>

1. B - Short explanation.
2. D - Short explanation.

</details>
```
