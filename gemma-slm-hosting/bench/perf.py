import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def _normalize_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prompts file: {path}")
    prompts: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            data = json.loads(line)
            if "id" not in data or "messages" not in data:
                raise ValueError("Prompt must include 'id' and 'messages'")
            prompts.append(data)
    return prompts


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d = k - f
    return values[f] + (values[c] - values[f]) * d


async def _fetch_first_model(
    client: httpx.AsyncClient, api_url: str, timeout: int
) -> str:
    response = await client.get(f"{api_url}/models", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    models = data.get("data", [])
    if not models:
        raise RuntimeError(f"No models returned from {api_url}/models")
    return models[0]["id"]


async def _run_perf(
    api_url: str,
    model: str,
    prompts: list[dict[str, Any]],
    total_requests: int,
    concurrency: int,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> tuple[list[float], int, int, float]:
    latencies: list[float] = []
    success_count = 0
    error_count = 0
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        start = time.perf_counter()

        async def one_request(idx: int) -> None:
            nonlocal success_count, error_count
            prompt = prompts[idx % len(prompts)]
            payload = {
                "model": model,
                "messages": prompt["messages"],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            async with sem:
                t0 = time.perf_counter()
                try:
                    response = await client.post(
                        f"{api_url}/chat/completions",
                        json=payload,
                        timeout=timeout,
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    async with lock:
                        if response.status_code == 200:
                            latencies.append(latency_ms)
                            success_count += 1
                        else:
                            error_count += 1
                except Exception:
                    async with lock:
                        error_count += 1

        tasks = [one_request(i) for i in range(total_requests)]
        await asyncio.gather(*tasks)
        total_elapsed = time.perf_counter() - start

    return latencies, success_count, error_count, total_elapsed


def _default_out_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("runs/perf") / f"perf_{stamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--prompts", default="data/prompts.jsonl")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    api_url = _normalize_url(args.url)
    prompts = _load_prompts(Path(args.prompts))
    out_path = Path(args.out) if args.out else _default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async def runner() -> None:
        async with httpx.AsyncClient() as client:
            model = args.model or await _fetch_first_model(
                client, api_url, args.timeout
            )
        latencies, success_count, error_count, total_elapsed = await _run_perf(
            api_url=api_url,
            model=model,
            prompts=prompts,
            total_requests=args.requests,
            concurrency=args.concurrency,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )

        latency_p50 = _percentile(latencies, 50)
        latency_p95 = _percentile(latencies, 95)
        throughput = success_count / total_elapsed if total_elapsed > 0 else 0.0

        summary = {
            "url": api_url,
            "model": model,
            "concurrency": args.concurrency,
            "total_requests": args.requests,
            "success_count": success_count,
            "error_count": error_count,
            "latency_ms_p50": latency_p50,
            "latency_ms_p95": latency_p95,
            "throughput_rps": throughput,
            "latencies_ms": latencies,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        print(f"url={api_url}")
        print(f"model={model}")
        print(f"concurrency={args.concurrency}")
        print(f"total_requests={args.requests}")
        print(f"success_count={success_count}")
        print(f"error_count={error_count}")
        print(f"latency_ms_p50={latency_p50:.2f}")
        print(f"latency_ms_p95={latency_p95:.2f}")
        print(f"throughput_rps={throughput:.2f}")

        out_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")

    asyncio.run(runner())


if __name__ == "__main__":
    main()
