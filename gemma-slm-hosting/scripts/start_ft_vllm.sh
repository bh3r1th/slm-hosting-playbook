#!/usr/bin/env bash
set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$script_dir/.env" ]; then
  set -a
  . "$script_dir/.env"
  set +a
fi

eval "$(python "$script_dir/read_pointer.py")"

: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${MAX_MODEL_LEN:=2048}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${DTYPE:=auto}"

if [ -z "$BASE_MODEL_ID" ]; then
  echo "Missing required BASE_MODEL_ID" >&2
  exit 1
fi
if [ -z "$ADAPTER_PATH" ]; then
  echo "Missing required ADAPTER_PATH" >&2
  exit 1
fi
if [ -z "$VLLM_HOST" ]; then
  echo "Missing required VLLM_HOST" >&2
  exit 1
fi
if [ -z "$VLLM_PORT_FT" ]; then
  echo "Missing required VLLM_PORT_FT" >&2
  exit 1
fi
if ! [[ "$VLLM_PORT_FT" =~ ^[0-9]+$ ]]; then
  echo "Invalid VLLM_PORT_FT='$VLLM_PORT_FT'" >&2
  exit 1
fi

if [ -z "$ADAPTER_PATH" ] || [ ! -d "$ADAPTER_PATH" ]; then
  echo "ADAPTER_PATH must be a local directory: $ADAPTER_PATH" >&2
  exit 1
fi

echo "Expected endpoints:"
echo "http://$VLLM_HOST:$VLLM_PORT_FT/v1/models"
echo "http://$VLLM_HOST:$VLLM_PORT_FT/v1/chat/completions"

python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL_ID" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT_FT" \
  --enable-lora \
  --lora-modules ft="$ADAPTER_PATH" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --dtype "$DTYPE" \
  --served-model-name "$BASE_MODEL_ID" \
  --served-model-name "ft"
