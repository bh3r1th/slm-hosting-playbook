#!/usr/bin/env bash
set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$script_dir/.env" ]; then
  set -a
  . "$script_dir/.env"
  set +a
fi

eval "$(python "$script_dir/read_pointer.py")"

if [ -z "$ADAPTER_PATH" ] || [ ! -d "$ADAPTER_PATH" ]; then
  echo "ADAPTER_PATH must be a local directory: $ADAPTER_PATH" >&2
  exit 1
fi

python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL_ID" \
  --port "$VLLM_PORT_FT" \
  --enable-lora \
  --lora-modules adapter="$ADAPTER_PATH" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --dtype "$DTYPE"
