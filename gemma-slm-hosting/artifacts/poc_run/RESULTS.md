# PoC Results Bundle

This folder captures one end-to-end PoC run with base and FT vLLM servers.

Contents:
- `base.json` / `ft.json`: deterministic chat outputs from the base and FT servers.
- `perf_*.json`: perf summaries for base and FT.
- `base.log` / `ft.log`: tail logs from each server run.
- `METADATA.txt`: run context (GPU, versions, model ids).

Quick checks:
- `bash scripts/pretty_ab.sh`
- `bash scripts/poc_run.sh`
