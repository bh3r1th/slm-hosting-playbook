import argparse
import json
import os
import time
from pathlib import Path

import requests


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((pct / 100.0) * (len(values) - 1)))
    return values[idx]


def _run(host: str, port: str, model: str, n: int) -> dict:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "temperature": 0.2,
        "max_tokens": 64,
    }
    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        _ = response.json()
        latencies.append(time.perf_counter() - start)
    return {
        "avg_latency_s": sum(latencies) / len(latencies),
        "p50_latency_s": _percentile(latencies, 50),
        "p95_latency_s": _percentile(latencies, 95),
        "samples": len(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-requests", type=int, default=50)
    args = parser.parse_args()

    host = os.getenv("VLLM_HOST", "127.0.0.1")
    port_base = os.getenv("VLLM_PORT_BASE", "8000")
    port_ft = os.getenv("VLLM_PORT_FT", "8001")
    base_model = os.getenv("BASE_MODEL_ID", "base")
    ft_model = os.getenv("FT_MODEL_ID", "adapter")

    results = {
        "base": _run(host, port_base, base_model, args.num_requests),
        "finetuned": _run(host, port_ft, ft_model, args.num_requests),
    }

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
