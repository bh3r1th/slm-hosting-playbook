#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${ROOT_DIR}/loadtest/reports"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

HOST="${GATEWAY_URL:-http://localhost:8000}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"
PROMPTS_FILE="${PROMPTS_FILE:-${ROOT_DIR}/loadtest/workloads/prompts_short.jsonl}"
USERS="${USERS:-10}"
SPAWN_RATE="${SPAWN_RATE:-2}"
RUN_TIME="${RUN_TIME:-30s}"

mkdir -p "${REPORT_DIR}"

PROMPTS_FILE="${PROMPTS_FILE}" ENDPOINT="${ENDPOINT}" \
  locust -f "${ROOT_DIR}/loadtest/locustfile.py" \
  --headless \
  -H "${HOST}" \
  -u "${USERS}" \
  -r "${SPAWN_RATE}" \
  -t "${RUN_TIME}" \
  --csv "${REPORT_DIR}/locust_${TIMESTAMP}" \
  --html "${REPORT_DIR}/locust_${TIMESTAMP}.html" \
  --logfile "${REPORT_DIR}/locust_${TIMESTAMP}.log" \
  --only-summary \
  --exit-code-on-error 1
