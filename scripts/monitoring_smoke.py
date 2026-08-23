"""
Smoke-test the local monitoring stack.

Checks:
- API health
- API Prometheus metrics endpoint
- Prometheus readiness and scrape target health
- Grafana health and dashboard provisioning
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _request_json(url: str, headers: dict[str, str] | None = None) -> Any:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_text(url: str, headers: dict[str, str] | None = None) -> str:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=10) as response:
        return response.read().decode("utf-8")


def _basic_auth(username: str, password: str) -> dict[str, str]:
    raw = f"{username}:{password}".encode()
    token = base64.b64encode(raw).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def check_api(api_url: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    health = _request_json(f"{api_url}/health")
    results.append(
        CheckResult(
            "api_health",
            health.get("status") == "healthy",
            f"status={health.get('status')} model_loaded={health.get('model_loaded')}",
        )
    )

    metrics = _request_text(f"{api_url}/metrics")
    required = [
        "inference_requests_total",
        "inference_latency_seconds",
        "model_inference_latency_seconds",
        "inference_errors_total",
    ]
    missing = [metric for metric in required if metric not in metrics]
    results.append(
        CheckResult(
            "api_metrics",
            not missing,
            "all required metrics present" if not missing else f"missing={missing}",
        )
    )
    return results


def check_prometheus(prometheus_url: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    ready = _request_text(f"{prometheus_url}/-/ready")
    results.append(CheckResult("prometheus_ready", "Ready" in ready, ready.strip()))

    targets = _request_json(f"{prometheus_url}/api/v1/targets?state=active")
    active_targets = targets["data"]["activeTargets"]
    target_health = {
        target["labels"]["job"]: target["health"] for target in active_targets
    }
    required_jobs = ["invoice-ner-app", "tritonserver"]
    down = {
        job: target_health.get(job, "missing")
        for job in required_jobs
        if target_health.get(job) != "up"
    }
    results.append(
        CheckResult(
            "prometheus_targets",
            not down,
            f"targets={target_health}" if not down else f"down={down}",
        )
    )
    return results


def check_grafana(grafana_url: str, username: str, password: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    headers = _basic_auth(username, password)
    health = _request_json(f"{grafana_url}/api/health")
    results.append(
        CheckResult(
            "grafana_health",
            health.get("database") == "ok",
            f"database={health.get('database')} version={health.get('version')}",
        )
    )

    query = urlencode({"query": "Invoice NER"})
    dashboards = _request_json(f"{grafana_url}/api/search?{query}", headers=headers)
    required_uids = {
        "invoice-ner-golden",
        "invoice-ner-executive",
        "invoice-ner-api-infra",
        "invoice-ner-model",
        "invoice-ner-load-test",
    }
    found_uids = {item.get("uid") for item in dashboards}
    missing_uids = sorted(required_uids - found_uids)
    results.append(
        CheckResult(
            "grafana_dashboards",
            not missing_uids,
            (
                f"indexed={sorted(required_uids)}"
                if not missing_uids
                else f"missing={missing_uids}"
            ),
        )
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test monitoring endpoints.")
    parser.add_argument("--api-url", default="http://localhost:7860")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--grafana-url", default="http://localhost:3000")
    parser.add_argument("--grafana-user", default="admin")
    parser.add_argument("--grafana-password", default="admin")
    args = parser.parse_args()

    checks: list[CheckResult] = []
    try:
        checks.extend(check_api(args.api_url.rstrip("/")))
        checks.extend(check_prometheus(args.prometheus_url.rstrip("/")))
        checks.extend(
            check_grafana(
                args.grafana_url.rstrip("/"),
                args.grafana_user,
                args.grafana_password,
            )
        )
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"monitoring smoke failed before completing all checks: {exc}")
        return 1

    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"{status} {check.name}: {check.detail}")

    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
