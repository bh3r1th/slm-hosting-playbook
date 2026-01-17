# Intent

Minimal scripts to launch Gemma vLLM servers and validate them quickly.
Includes a base model path and an optional LoRA adapter path.

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
