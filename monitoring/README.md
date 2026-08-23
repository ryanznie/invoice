# Monitoring Runbook

## Start the Stack

Create a local `.env` file before starting the stack:

```bash
cp .env.example .env
```

The example password is only for localhost development. Change
`GRAFANA_ADMIN_PASSWORD` in `.env` before sharing access to Grafana.

```bash
docker compose up -d --build invoice-ner tritonserver prometheus grafana
```

Grafana binds to `127.0.0.1:3000` by default. Set `GRAFANA_HOST=0.0.0.0`
only when the Docker host is protected by trusted network controls.

## Smoke Test

```bash
uv run python scripts/monitoring_smoke.py
```

Checks API health, API metrics, Prometheus readiness, Prometheus scrape targets,
Grafana health, and all provisioned `Invoice NER` dashboards.

## Production SLOs

Production SLOs, alert policy, error budget, eval gates, and hardening backlog
live in:

```text
docs/PRODUCTION_MONITORING.md
```

## Build Grafana Dashboards

Generate the provisioned dashboard JSON files:

```bash
uv run python scripts/generate_grafana_dashboards.py
```

Grafana loads every JSON file from:

```text
monitoring/grafana/dashboards/
```

The dashboards are mounted into the container at `/var/lib/grafana/dashboards`
and provisioned by:

```text
monitoring/grafana/provisioning/dashboards/dashboard.yml
```

After generation, restart Grafana or wait for the provisioning poll:

```bash
docker compose restart grafana
```

Open:

- `http://localhost:3000/d/invoice-ner-executive`
- `http://localhost:3000/d/invoice-ner-api-infra`
- `http://localhost:3000/d/invoice-ner-model`
- `http://localhost:3000/d/invoice-ner-load-test`

## Locust Load Test

Interactive UI:

```bash
uv run locust -f locustfile.py --host=http://localhost:7860
```

Headless run with CSV output:

```bash
uv run locust -f locustfile.py \
  --host=http://localhost:7860 \
  --headless \
  --users=10 \
  --spawn-rate=2 \
  --run-time=60s \
  --csv monitoring/locust/invoice_ner
```

## Offline Extraction Eval

```bash
uv run python scripts/eval_invoice_extraction.py \
  --api-url http://localhost:7860 \
  --dataset data/train/qa_dataset.json \
  --image-root data/SROIE2019/train/img \
  --limit 25 \
  --output-dir monitoring/evals/latest
```

Tracks exact match, normalized exact match, edit distance, valid-format rate,
not-found rate, extraction method counts, and latency summary.
