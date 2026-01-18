# Intent

Minimal scripts to launch Gemma vLLM servers and validate them quickly.
Includes a base model path and an optional LoRA adapter path.

# PoC proof (end-to-end)

- Proves: LoRA adapter from Phase 1 loads in vLLM and serves via OpenAI-compatible endpoints.
- Proves: smoke tests pass and A/B outputs + basic perf are captured for base and FT.
- Proves: sequential base → FT run works under tight VRAM limits.

Requires an NVIDIA GPU; Colab Pro GPU recommended. Base + FT run sequentially due to VRAM constraints.

Repro steps (commands only, `.env` already set):
- `make setup`
- `make start-base && make smoke MODE=base && make stop`
- `make start-ft && make smoke MODE=ft && make stop`
- `make ab MODE=seq`
- `make perf MODE=seq`
- `bash scripts/poc_run.sh`

Evidence bundle: `artifacts/poc_run/`
- `base.json` / `ft.json`: deterministic sample outputs
- `perf_*.json`: perf summaries
- `base.log` / `ft.log`: tail logs

# Setup

- copy `.env.example` -> `.env`
- do not commit `.env`
- set `ADAPTER_PATH` (local path OR Drive path in Colab)

# Colab Run (Minimal)

- mount Drive
- Runtime → Change runtime type → GPU
- copy `.env.example` -> `.env` and set `ADAPTER_PATH`
- if OOM, reduce `MAX_NUM_SEQS`, `MAX_MODEL_LEN`, or `GPU_MEMORY_UTILIZATION` in `.env`
- `make setup`
- `make start-both`
- `make health`
- `make smoke MODE=both`
- `make ab PROMPTS=data/prompts.jsonl OUT=runs/ab`
- `make perf`

# Run base

```bash
bash scripts/start_base_vllm.sh
```

# Run finetuned

```bash
bash scripts/start_ft_vllm.sh
```

# Smoke test

```bash
python scripts/smoke_test.py
```

# Benchmark

```bash
python scripts/benchmark.py --n 50
```

# A/B Eval

```bash
python eval/ab_eval.py --prompts data/prompts.jsonl --out-dir runs/ab
```
