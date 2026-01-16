#!/usr/bin/env bash
# This script starts vLLM in OpenAI-compatible mode (v1/* endpoints) for local testing.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$script_dir/.env" ]; then
  declare -A env_overrides=()
  while IFS= read -r line; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    if [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)= ]]; then
      key="${BASH_REMATCH[2]}"
      if printenv "$key" >/dev/null 2>&1; then
        env_overrides["$key"]="$(printenv "$key")"
      fi
    fi
  done < "$script_dir/.env"
  set -a
  # shellcheck disable=SC1090
  . "$script_dir/.env"
  set +a
  for key in "${!env_overrides[@]}"; do
    export "$key=${env_overrides[$key]}"
  done
fi

eval "$(python "$script_dir/read_pointer.py")"

: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${MAX_MODEL_LEN:=2048}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${DTYPE:=auto}"

if [ -z "${BASE_MODEL_ID:-}" ]; then
  echo "Missing required BASE_MODEL_ID" >&2
  exit 1
fi
if [ -z "${VLLM_HOST:-}" ]; then
  echo "Missing required VLLM_HOST" >&2
  exit 1
fi
if [ -z "${VLLM_PORT_BASE:-}" ]; then
  echo "Missing required VLLM_PORT_BASE" >&2
  exit 1
fi
if ! [[ "$VLLM_PORT_BASE" =~ ^[0-9]+$ ]]; then
  echo "Invalid VLLM_PORT_BASE='$VLLM_PORT_BASE'" >&2
  exit 1
fi

echo "Expected endpoints:"
echo "http://$VLLM_HOST:$VLLM_PORT_BASE/v1/models"
echo "http://$VLLM_HOST:$VLLM_PORT_BASE/v1/chat/completions"

python -m vllm.entrypoints.openai.api_server \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT_BASE" \
  --model "$BASE_MODEL_ID" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --dtype "$DTYPE"
