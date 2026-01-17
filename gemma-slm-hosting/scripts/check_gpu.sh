#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; GPU required" >&2
  exit 1
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "No GPU detected by nvidia-smi" >&2
  exit 1
fi

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr -d '\r')"
gpu_mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1 | tr -d '\r')"
echo "GPU: $gpu_name"
echo "VRAM: $gpu_mem"
