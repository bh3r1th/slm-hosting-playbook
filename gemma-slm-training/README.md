# Phase 1 — Gemma SLM Training (Blog Part 1)

This folder covers **fine-tuning Gemma** on a public dataset and producing **versioned artifacts** that can be served in Phase 2.

## Outputs from this phase

- Adapter artifacts (LoRA/QLoRA)
- Evaluation results
- Optional: merged/exported model artifacts for serving

## Where artifacts go

- Training outputs: `artifacts/`
- Metrics/results: `results/`

## Install

```bash
uv venv && uv pip install -r requirements.txt
# OR
uv pip install -e .
```

## Quickstart

```bash
cp .env.example .env
python scripts/smoke_model_access.py
python scripts/prepare_data.py --max-train 50000 --max-val 2000
python scripts/train_lora.py --qlora --epochs 1 --max-seq-len 1024
python scripts/export_artifacts.py
```

## Next step

Proceed to Phase 2 (hosting):

- [`../gemma-slm-hosting/`](../gemma-slm-hosting/)
