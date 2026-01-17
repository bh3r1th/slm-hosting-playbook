#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

if [ ! -f "$repo_root/.env" ]; then
  echo "Missing .env at $repo_root/.env" >&2
  exit 1
fi

# shellcheck source=lib_env.sh
. "$script_dir/lib_env.sh"
load_env

if [ -z "${BASE_API_URL:-}" ]; then
  echo "Missing BASE_API_URL" >&2
  exit 1
fi
if [ -z "${FT_API_URL:-}" ]; then
  echo "Missing FT_API_URL" >&2
  exit 1
fi
require_vars BASE_API_URL FT_API_URL

BASE_API_URL="${BASE_API_URL%/}"
FT_API_URL="${FT_API_URL%/}"

payload='{"model":"__MODEL__","messages":[{"role":"user","content":"Say hi."}],"temperature":0,"max_tokens":16}'

check_endpoint() {
  local name="$1"
  local api_url="$2"
  local model="$3"
  local ok=1

  local status
  status="$(curl -sS -o /dev/null -w "%{http_code}" "$api_url/v1/models" || true)"
  if [ "$status" != "200" ]; then
    echo "FAIL $name models $status"
    return 1
  fi

  local body
  body="${payload/__MODEL__/$model}"
  status="$(curl -sS -o /dev/null -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d "$body" \
    "$api_url/v1/chat/completions" || true)"
  if [ "$status" != "200" ]; then
    echo "FAIL $name chat $status"
    return 1
  fi

  echo "PASS $name"
  return 0
}

base_model="${BASE_MODEL_ID:-}"
ft_model="${FT_SERVED_MODEL_NAME:-ft}"

base_ok=0
ft_ok=0

if check_endpoint "BASE" "$BASE_API_URL" "$base_model"; then
  base_ok=1
fi
if check_endpoint "FT" "$FT_API_URL" "$ft_model"; then
  ft_ok=1
fi

if [ "$base_ok" -ne 1 ] || [ "$ft_ok" -ne 1 ]; then
  exit 1
fi
