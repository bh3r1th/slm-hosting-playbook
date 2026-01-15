# Intent

Minimal scripts to launch Gemma vLLM servers and validate them quickly.
Includes a base model path and an optional LoRA adapter path.

# Setup

- copy `.env.example` -> `.env`
- do not commit `.env`
- set `HF_TOKEN`
- set `ADAPTER_PATH` (local path OR Drive path in Colab)

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
