#!/usr/bin/env bash
set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8000}"

curl -fsS "${GATEWAY_URL}/health" | cat
