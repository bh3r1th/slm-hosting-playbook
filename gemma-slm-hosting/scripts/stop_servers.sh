#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
pid_dir="$repo_root/runs/pids"

# shellcheck source=lib_env.sh
. "$script_dir/lib_env.sh"
load_env

warn() {
  echo "WARN: $*" >&2
}

stop_pid() {
  local name="$1"
  local pid_file="$pid_dir/$name.pid"
  if [ ! -f "$pid_file" ]; then
    return 2
  fi
  local pid
  pid="$(cat "$pid_file" || true)"
  if [ -z "$pid" ]; then
    echo "Empty pid file for $name"
    rm -f "$pid_file"
    return 0
  fi
  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "Stopping $name (pid $pid)"
    kill "$pid" || true
    for _ in {1..10}; do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        rm -f "$pid_file"
        return 0
      fi
      sleep 0.5
    done
    return 1
  fi
  echo "Process not running for $name (pid $pid)"
  rm -f "$pid_file"
  return 0
}

stop_by_port() {
  local name="$1"
  local port="$2"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti tcp:"$port" || true)"
  elif command -v fuser >/dev/null 2>&1; then
    pids="$(fuser -n tcp "$port" 2>/dev/null || true)"
  else
    warn "No lsof/fuser to stop $name by port $port"
    return 2
  fi
  if [ -z "$pids" ]; then
    return 2
  fi
  echo "Stopping $name on port $port (pid(s): $pids)"
  kill $pids || true
  sleep 1
  for pid in $pids; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      return 1
    fi
  done
  return 0
}

fail=0

status=0
stop_pid "base" || status=$?
if [ "$status" -eq 1 ]; then
  fail=1
elif [ "$status" -eq 2 ]; then
  base_port="${BASE_PORT:-${VLLM_PORT_BASE:-}}"
  if [ -n "$base_port" ]; then
    if ! stop_by_port "base" "$base_port"; then
      fail=1
    fi
  else
    warn "BASE_PORT not set; cannot stop base by port"
  fi
fi

status=0
stop_pid "ft" || status=$?
if [ "$status" -eq 1 ]; then
  fail=1
elif [ "$status" -eq 2 ]; then
  ft_port="${FT_PORT:-${VLLM_PORT_FT:-}}"
  if [ -n "$ft_port" ]; then
    if ! stop_by_port "ft" "$ft_port"; then
      fail=1
    fi
  else
    warn "FT_PORT not set; cannot stop ft by port"
  fi
fi

if [ "$fail" -ne 0 ]; then
  exit 1
fi
