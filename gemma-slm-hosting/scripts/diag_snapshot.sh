#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
root_out="$repo_root/runs/diag"

warn() {
  echo "WARN: $*" >&2
}

stamp="$(date +%Y%m%d-%H%M%S)"
out_dir="$root_out/$stamp"

mkdir -p "$out_dir"

# shellcheck source=lib_env.sh
. "$script_dir/lib_env.sh"
load_env
if [ ! -f "$repo_root/.env" ]; then
  warn "Missing .env at $repo_root/.env"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >"$out_dir/nvidia_smi.txt" 2>&1 || warn "nvidia-smi failed"
else
  warn "nvidia-smi not found"
fi

python --version >"$out_dir/python_version.txt" 2>&1 || warn "python --version failed"
python -m pip freeze >"$out_dir/pip_freeze.txt" 2>&1 || warn "pip freeze failed"

df -h >"$out_dir/disk_df_h.txt" 2>&1 || warn "df -h failed"
if command -v free >/dev/null 2>&1; then
  free -h >"$out_dir/mem_free_h.txt" 2>&1 || warn "free -h failed"
else
  warn "free not found"
fi

ps aux | grep -E 'vllm|python' | grep -v -E 'grep' >"$out_dir/processes_vllm.txt" 2>&1 || warn "ps aux failed"

if command -v ss >/dev/null 2>&1; then
  ss -lntp >"$out_dir/ports_listening.txt" 2>&1 || warn "ss -lntp failed"
elif command -v netstat >/dev/null 2>&1; then
  netstat -tulpn >"$out_dir/ports_listening.txt" 2>&1 || warn "netstat -tulpn failed"
else
  warn "ss/netstat not found"
fi

{
  echo "BASE_MODEL_ID=${BASE_MODEL_ID:-}"
  echo "ADAPTER_PATH=${ADAPTER_PATH:-}"
  echo "HOST=${HOST:-}"
  echo "BASE_PORT=${BASE_PORT:-}"
  echo "FT_PORT=${FT_PORT:-}"
  echo "BASE_API_URL=${BASE_API_URL:-}"
  echo "FT_API_URL=${FT_API_URL:-}"
  echo "DTYPE=${DTYPE:-}"
  echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-}"
  echo "MAX_MODEL_LEN=${MAX_MODEL_LEN:-}"
  echo "MAX_NUM_SEQS=${MAX_NUM_SEQS:-}"
  echo "ENFORCE_EAGER=${ENFORCE_EAGER:-}"
} >"$out_dir/env_selected.txt" 2>&1 || warn "env_selected.txt failed"

if [ -f "$repo_root/runs/logs/base.log" ]; then
  tail -n 200 "$repo_root/runs/logs/base.log" >"$out_dir/logs_base_tail.txt" 2>&1 || warn "base log tail failed"
else
  warn "base log not found"
fi

if [ -f "$repo_root/runs/logs/ft.log" ]; then
  tail -n 200 "$repo_root/runs/logs/ft.log" >"$out_dir/logs_ft_tail.txt" 2>&1 || warn "ft log tail failed"
else
  warn "ft log not found"
fi

tar -czf "$root_out/$stamp.tar.gz" -C "$root_out" "$stamp" >/dev/null 2>&1 || warn "tar failed"

echo "Snapshot: $out_dir"
