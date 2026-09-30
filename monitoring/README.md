# Monitoring Runbook

The local stack includes the API, Triton, Prometheus, and Grafana. Compose requires a .env file with a non-empty GRAFANA_ADMIN_PASSWORD.

    cp .env.example .env

Set a strong Grafana password in .env, then start the stack:

    docker compose up -d --build invoice-ner tritonserver prometheus grafana

Grafana is bound to localhost:3000 by default. Keep it private; only bind it to another interface behind trusted network controls.

## Checks

    uv run python scripts/monitoring_smoke.py

The smoke test checks API health and metrics, Prometheus readiness and targets, Grafana health, and provisioned dashboards.

To regenerate dashboards and restart Grafana:

    uv run python scripts/generate_grafana_dashboards.py
    docker compose restart grafana

## Load and quality checks

Interactive Locust:

    uv run locust -f tests/load/locustfile.py --host=http://localhost:7860

Offline extraction evaluation:

    uv run python scripts/eval_invoice_extraction.py --api-url http://localhost:7860 --dataset data/train/qa_dataset.json --image-root data/SROIE2019/train/img --limit 25 --output-dir monitoring/evals/latest

These checks need a running API and the corresponding local dataset. See [Production Monitoring](../docs/PRODUCTION_MONITORING.md) for SLOs, alert policy, and full runbook.
