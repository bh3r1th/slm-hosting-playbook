# slm-hosting-playbook

A practical, open-source playbook for learning and benchmarking **self-hosted Small Language Model (SLM) serving** end-to-end.

This repo is intentionally **hands-on**: it focuses on building a runnable serving stack, measuring real latency (P50/P95/P99), stress testing under load, and packaging everything so it’s reproducible. No fluff.

## What’s in this repo

### Projects
- **`gemma-slm-hosting/`** — A complete starter stack for hosting **Google Gemma** with:
  - model serving (vLLM)
  - a simple API gateway
  - observability (Prometheus + Grafana + tracing)
  - load testing harness (Locust)
  - scripts + configs to run benchmarks and capture results

## Why this repo exists

Most “self-hosting” content stops at “it runs.” This playbook is about the next step:
- **Does it stay up under bursty traffic?**
- **What happens to P95/P99 and TTFT when prompts get longer?**
- **Which knobs actually move tail latency and throughput?**
- **How do you make the setup reproducible for others?**

## How to use it

Start here:
- Go to **[`gemma-slm-hosting/`](./gemma-slm-hosting/)** and follow its README.

As you add more SLM hosting projects (different models / runtimes / hardware), this repo becomes your consolidated “playbook” with repeatable patterns.

## Guiding principles

- **Open-source first**
- **Measure everything that matters** (tail latency, TTFT, throughput, error rate)
- **Reproducible experiments** (scripts + configs + pinned deps)
- **Document what breaks** (gotchas > marketing)

## License

See [LICENSE](./LICENSE).
