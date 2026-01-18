#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
art_dir="$repo_root/artifacts/poc_run"

# shellcheck source=lib_env.sh
. "$script_dir/lib_env.sh"
load_env

mkdir -p "$art_dir"
cd "$repo_root"

base_port="${BASE_PORT:-8000}"
ft_port="${FT_PORT:-8001}"
base_api="${BASE_API_URL:-http://127.0.0.1:$base_port}"
ft_api="${FT_API_URL:-http://127.0.0.1:$ft_port}"
base_model="${BASE_MODEL_ID:-google/gemma-3-1b-it}"
ft_model="${SERVED_MODEL_NAME:-${FT_SERVED_MODEL_NAME:-ft}}"

if [ -z "${BASE_MODEL_ID:-}" ]; then
  echo "WARN: BASE_MODEL_ID not set; using default $base_model" >&2
fi

wait_ready() {
  local url="$1"
  local deadline=$((SECONDS + 600))
  while [ "$SECONDS" -lt "$deadline" ]; do
    code="$(curl -sS -o /dev/null -w "%{http_code}" "$url" || true)"
    if [ "$code" = "200" ]; then
      return 0
    fi
    sleep 5
  done
  echo "Timeout waiting for $url" >&2
  return 1
}

write_metadata() {
  {
    echo "date=$(date)"
    echo "hostname=$(hostname)"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    else
      echo "nvidia-smi=missing"
    fi
    python --version 2>&1 || true
    python -c "import vllm; print(vllm.__version__)" 2>/dev/null || true
    echo "BASE_MODEL_ID=${BASE_MODEL_ID:-}"
    echo "ADAPTER_PATH=${ADAPTER_PATH:-}"
  } >"$art_dir/METADATA.txt"
}

make stop || true
make start-base
wait_ready "$base_api/v1/models"
python scripts/smoke_test.py --mode base
curl -sS -X POST "$base_api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$base_model\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain LoRA in one sentence, then give one example use-case.\"}],\"temperature\":0,\"max_tokens\":80}" \
  >"$art_dir/base.json"
if [ -f "runs/logs/base.log" ]; then
  tail -n 200 "runs/logs/base.log" >"$art_dir/base.log"
fi
make stop || true

if [ -z "${ADAPTER_PATH:-}" ]; then
  echo "ERROR: ADAPTER_PATH not set; FT run cannot proceed" >&2
  exit 1
fi

make start-ft
wait_ready "$ft_api/v1/models"
python scripts/smoke_test.py --mode ft
curl -sS -X POST "$ft_api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$ft_model\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain LoRA in one sentence, then give one example use-case.\"}],\"temperature\":0,\"max_tokens\":80}" \
  >"$art_dir/ft.json"
if [ -f "runs/logs/ft.log" ]; then
  tail -n 200 "runs/logs/ft.log" >"$art_dir/ft.log"
fi
make stop || true

write_metadata

echo "DONE"
ls -1 "$art_dir"
