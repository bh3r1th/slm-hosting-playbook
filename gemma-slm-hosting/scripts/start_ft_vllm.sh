#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
log_dir="$repo_root/runs/logs"
pid_dir="$repo_root/runs/pids"
log_file="$log_dir/ft.log"
pid_file="$pid_dir/ft.pid"

# shellcheck source=lib_env.sh
. "$script_dir/lib_env.sh"
load_env
if [ -f "$repo_root/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$repo_root/.env"
  set +a
fi
if [ ! -f "$repo_root/.env" ]; then
  echo "Missing .env at $repo_root/.env. Run: make env" >&2
fi

eval "$(python "$script_dir/read_pointer.py")"

: "${HOST:=}"
: "${FT_PORT:=}"

mkdir -p "$log_dir" "$pid_dir"

HOST="${HOST:-${VLLM_HOST:-}}"
FT_PORT="${FT_PORT:-${VLLM_PORT_FT:-}}"
FT_API_URL="${FT_API_URL:-http://$HOST:$FT_PORT}"

: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${MAX_MODEL_LEN:=2048}"
: "${MAX_NUM_SEQS:=128}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${DTYPE:=auto}"
: "${ENFORCE_EAGER:=0}"

require_vars BASE_MODEL_ID ADAPTER_PATH HOST FT_PORT FT_API_URL

if [ -z "${HOST:-}" ]; then
  echo "Missing required HOST" >&2
  exit 1
fi
if [ -z "${FT_PORT:-}" ]; then
  echo "Missing required FT_PORT" >&2
  exit 1
fi
if [ -z "${BASE_MODEL_ID:-}" ]; then
  echo "Missing required BASE_MODEL_ID" >&2
  exit 1
fi
if [ -z "${ADAPTER_PATH:-}" ]; then
  echo "Missing required ADAPTER_PATH" >&2
  exit 1
fi
if [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
  echo "Missing adapter file: $ADAPTER_PATH/adapter_config.json" >&2
  exit 1
fi
if [ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]; then
  echo "Missing adapter file: $ADAPTER_PATH/adapter_model.safetensors" >&2
  exit 1
fi
if ! [[ "$FT_PORT" =~ ^[0-9]+$ ]]; then
  echo "Invalid FT_PORT='$FT_PORT'" >&2
  exit 1
fi

echo "Starting FT vLLM"
echo "HOST=$HOST"
echo "PORT=$FT_PORT"
echo "BASE_MODEL_ID=$BASE_MODEL_ID"
echo "ADAPTER_PATH=$ADAPTER_PATH"

echo "Expected endpoints:"
echo "http://$HOST:$FT_PORT/v1/models"
echo "http://$HOST:$FT_PORT/v1/chat/completions"

extra_args=()
if [[ "${ENFORCE_EAGER}" == "1" || "${ENFORCE_EAGER}" == "true" ]]; then
  extra_args+=(--enforce-eager)
fi

python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL_ID" \
  --host "$HOST" \
  --port "$FT_PORT" \
  --enable-lora \
  --lora-modules ft="$ADAPTER_PATH" \
  --served-model-name ft \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --dtype "$DTYPE" \
  "${extra_args[@]}" \
  >"$log_file" 2>&1 &

echo $! >"$pid_file"
echo "Started FT vLLM (pid $(cat "$pid_file"))"
echo "Logs: $log_file"
