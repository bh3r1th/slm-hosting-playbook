#!/usr/bin/env bash

load_env() {
  local script_dir repo_root env_file line key value
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "$script_dir/.." && pwd)"
  env_file="$repo_root/.env"
  if [ ! -f "$env_file" ]; then
    return 0
  fi
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    line="${line#export }"
    case "$line" in
      [A-Za-z_][A-Za-z0-9_]*=*)
        key="${line%%=*}"
        value="${line#*=}"
        export "$key=$value"
        ;;
    esac
  done < "$env_file"
}

require_vars() {
  local missing=0 var_name
  for var_name in "$@"; do
    if [ -z "${!var_name:-}" ]; then
      echo "Missing required env var: $var_name" >&2
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    exit 2
  fi
}
