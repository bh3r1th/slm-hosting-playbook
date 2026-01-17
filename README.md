# slm-hosting-playbook

A practical, open-source playbook for learning and benchmarking **self-hosted Small Language Model (SLM) serving** end-to-end — including **training + hosting**.

This repo is intentionally hands-on: build, measure, stress test, and package everything so it’s reproducible. No fluff.

## GPU required

Training (LoRA/QLoRA) and hosting (vLLM) require an NVIDIA GPU; CPU-only is not practical.
In Colab you will typically get T4, L4, A100, or H100; VRAM limits can cause OOM during vLLM warmup or higher concurrency.
If you hit OOM, reduce `max_num_seqs`, `max_model_len`, or `gpu_memory_utilization`.
If you run outside Colab, Docker GPU support is also required.

## Minimal end-to-end PoC

Train in `gemma-slm-training/`.
Export adapter artifacts (`adapter_config.json` + `adapter_model.safetensors`).
Store adapters locally or mount from Google Drive in Colab.
Start base and FT vLLM servers in `gemma-slm-hosting/`.
Hit the OpenAI-compatible endpoints.
Run smoke tests, then A/B eval and perf to compare base vs FT.

## Quick links

- `gemma-slm-training/`
- `gemma-slm-hosting/`
- `gemma-slm-hosting/README.md`
- `gemma-slm-hosting/runs/`

## Phase-based approach

### Phase 1 — Training (Blog Part 1)
Fine-tune a Gemma checkpoint on a public dataset and produce versioned model artifacts ready for serving.

📁 See: [`gemma-slm-training/`](./gemma-slm-training/)

### Phase 2 — Hosting & Benchmarking (vLLM) (Blog Part 2)
Serve the base Gemma model and the fine-tuned model, add observability, and run load tests to measure tail latency and throughput.

📁 See: [`gemma-slm-hosting/`](./gemma-slm-hosting/)

## Repo structure

- `gemma-slm-training/` — training + evaluation + export artifacts (Part 1)
- `gemma-slm-hosting/` — serving + observability + load testing (Part 2)

## License

MIT License. See [LICENSE](./LICENSE).

