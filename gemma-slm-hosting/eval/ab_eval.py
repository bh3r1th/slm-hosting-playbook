import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


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


def _fetch_first_model(client: httpx.Client, api_url: str, timeout: int) -> str:
    response = client.get(f"{api_url}/models", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    models = data.get("data", [])
    if not models:
        raise RuntimeError(f"No models returned from {api_url}/models")
    return models[0]["id"]


def _chat(
    client: httpx.Client,
    api_url: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    try:
        response = client.post(
            f"{api_url}/chat/completions",
            json=payload,
            timeout=timeout,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        status = response.status_code
        data = response.json() if response.content else {}
        text = ""
        usage = None
        if status == 200:
            choices = data.get("choices", [])
            if choices:
                text = choices[0]["message"]["content"]
            usage = data.get("usage")
        return {
            "model": model,
            "latency_ms": latency_ms,
            "status": status,
            "text": text,
            "usage": usage,
        }
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "model": model,
            "latency_ms": latency_ms,
            "status": None,
            "error": str(exc),
        }


def _resolve_out_path(out: str | None, out_dir: str | None) -> Path:
    if out and out_dir:
        raise ValueError("Use --out or --out-dir, not both")
    if out:
        return Path(out)
    if not out_dir:
        out_dir = "runs/ab"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path(out_dir) / f"ab_{stamp}.jsonl"


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="data/prompts.jsonl")
    parser.add_argument("--out")
    parser.add_argument("--out-dir")
    parser.add_argument("--base-url", default=os.getenv("BASE_API_URL", ""))
    parser.add_argument("--ft-url", default=os.getenv("FT_API_URL", ""))
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    out_path = _resolve_out_path(args.out, args.out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_url = args.base_url.rstrip("/")
    ft_url = args.ft_url.rstrip("/")
    if not base_url or not ft_url:
        raise ValueError("BASE_API_URL and FT_API_URL must be set")

    prompts = _load_prompts(prompts_path)
    iterator = tqdm(prompts, desc="A/B") if tqdm else prompts

    with httpx.Client() as client, out_path.open("w", encoding="utf-8") as out:
        if args.model:
            base_model = args.model
            ft_model = args.model
        else:
            base_model = _fetch_first_model(client, base_url, args.timeout)
            ft_model = _fetch_first_model(client, ft_url, args.timeout)
        for prompt in iterator:
            prompt_id = str(prompt["id"])
            messages = prompt["messages"]
            ts = datetime.now(timezone.utc).isoformat()
            base_result = _chat(
                client,
                base_url,
                base_model,
                messages,
                args.temperature,
                args.max_tokens,
                args.timeout,
            )
            ft_result = _chat(
                client,
                ft_url,
                ft_model,
                messages,
                args.temperature,
                args.max_tokens,
                args.timeout,
            )
            record = {
                "prompt_id": prompt_id,
                "request": {
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                },
                "base": base_result,
                "ft": ft_result,
                "timestamp": ts,
            }
            out.write(json.dumps(record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
