#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

# shellcheck source=lib_env.sh
. "$script_dir/lib_env.sh"
load_env

fail=0

check() {
  local label="$1"
  local ok="$2"
  if [ "$ok" -eq 1 ]; then
    echo "PASS $label"
  else
    echo "FAIL $label"
    fail=1
  fi
}

if [ -f "$repo_root/.env" ]; then
  echo "PASS .env exists"
else
  echo "WARN .env missing"
fi

if [ -n "${BASE_MODEL_ID:-}" ]; then
  check "BASE_MODEL_ID set" 1
else
  check "BASE_MODEL_ID set" 0
fi

adapter_ok=1
if [ -z "${ADAPTER_PATH:-}" ]; then
  adapter_ok=0
else
  if [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
    adapter_ok=0
  fi
  if [ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]; then
    adapter_ok=0
  fi
fi
check "ADAPTER_PATH adapter files" "$adapter_ok"

base_port_ok=1
ft_port_ok=1
if ! [[ "${BASE_PORT:-}" =~ ^[0-9]+$ ]]; then
  base_port_ok=0
fi
if ! [[ "${FT_PORT:-}" =~ ^[0-9]+$ ]]; then
  ft_port_ok=0
fi
check "BASE_PORT integer" "$base_port_ok"
check "FT_PORT integer" "$ft_port_ok"

if command -v curl >/dev/null 2>&1; then
  check "curl available" 1
else
  check "curl available" 0
fi

if [[ "${ADAPTER_PATH:-}" == /content/drive* ]]; then
  if [ -d "/content/drive" ]; then
    check "Colab Drive mounted" 1
  else
    check "Colab Drive mounted" 0
  fi
fi

if [ "$fail" -ne 0 ]; then
  exit 1
fi
