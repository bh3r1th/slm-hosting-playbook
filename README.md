# slm-hosting-playbook

A practical, open-source playbook for learning and benchmarking **self-hosted Small Language Model (SLM) serving** end-to-end — including **training + hosting**.

This repo is intentionally hands-on: build, measure, stress test, and package everything so it’s reproducible. No fluff.

## Phase-based approach

### Phase 1 — Training (Blog Part 1)
Fine-tune a Gemma checkpoint on a public dataset and produce versioned model artifacts ready for serving.

📁 See: [`gemma-slm-training/`](./gemma-slm-training/)

### Phase 2 — Hosting & Benchmarking (Blog Part 2)
Serve the base Gemma model and the fine-tuned model, add observability, and run load tests to measure tail latency and throughput.

📁 See: [`gemma-slm-hosting/`](./gemma-slm-hosting/)

### Phase 2: Hosting (vLLM)
See: [gemma-slm-hosting/README.md](./gemma-slm-hosting/README.md)

## Repo structure

- `gemma-slm-training/` — training + evaluation + export artifacts (Part 1)
- `gemma-slm-hosting/` — serving + observability + load testing (Part 2)

## License

See [LICENSE](./LICENSE).

