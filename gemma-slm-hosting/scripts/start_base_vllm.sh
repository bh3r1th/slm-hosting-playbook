#!/usr/bin/env bash
# This script starts vLLM in OpenAI-compatible mode (v1/* endpoints) for local testing.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
log_dir="$repo_root/runs/logs"
pid_dir="$repo_root/runs/pids"
log_file="$log_dir/base.log"
pid_file="$pid_dir/base.pid"

# shellcheck source=lib_env.sh
. "$script_dir/lib_env.sh"
load_env

eval "$(python "$script_dir/read_pointer.py")"

: "${HOST:=}"
: "${BASE_PORT:=}"

mkdir -p "$log_dir" "$pid_dir"

HOST="${HOST:-${VLLM_HOST:-}}"
BASE_PORT="${BASE_PORT:-${VLLM_PORT_BASE:-}}"
BASE_API_URL="${BASE_API_URL:-http://$HOST:$BASE_PORT}"

: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${MAX_MODEL_LEN:=2048}"
: "${MAX_NUM_SEQS:=128}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${DTYPE:=auto}"
: "${ENFORCE_EAGER:=0}"

require_vars BASE_MODEL_ID HOST BASE_PORT BASE_API_URL

if [ -z "${BASE_MODEL_ID:-}" ]; then
  echo "Missing required BASE_MODEL_ID" >&2
  exit 1
fi
if [ -z "${HOST:-}" ]; then
  echo "Missing required HOST" >&2
  exit 1
fi
if [ -z "${BASE_PORT:-}" ]; then
  echo "Missing required BASE_PORT" >&2
  exit 1
fi
if ! [[ "$BASE_PORT" =~ ^[0-9]+$ ]]; then
  echo "Invalid BASE_PORT='$BASE_PORT'" >&2
  exit 1
fi

echo "Expected endpoints:"
echo "http://$HOST:$BASE_PORT/v1/models"
echo "http://$HOST:$BASE_PORT/v1/chat/completions"

extra_args=()
if [[ "${ENFORCE_EAGER}" == "1" || "${ENFORCE_EAGER}" == "true" ]]; then
  extra_args+=(--enforce-eager)
fi

python -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$BASE_PORT" \
  --model "$BASE_MODEL_ID" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --dtype "$DTYPE" \
  "${extra_args[@]}" \
  >"$log_file" 2>&1 &

echo $! >"$pid_file"
echo "Started base vLLM (pid $(cat "$pid_file"))"
echo "Logs: $log_file"
