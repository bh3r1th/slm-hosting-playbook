import os
import sys
import time
from pathlib import Path
import argparse

import requests
from dotenv import load_dotenv


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _wait_ready(host: str, port: int, timeout_seconds: int) -> None:
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server not ready at {url} after {timeout_seconds}s")


def _post_chat(host: str, port: int, model_name: str, prompt: str) -> str:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 128,
    }
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "ft", "both"], default="both")
    parser.add_argument(
        "--host", default=os.getenv("VLLM_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=int(os.getenv("VLLM_PORT_BASE", "8000")),
    )
    parser.add_argument(
        "--port-ft",
        type=int,
        default=int(os.getenv("VLLM_PORT_FT", "8001")),
    )
    parser.add_argument(
        "--model", default=os.getenv("BASE_MODEL_ID", "google/gemma-3-1b-it")
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.getenv("TIMEOUT_SECONDS", "180")),
    )
    args = parser.parse_args()

    host = args.host
    port_base = args.port_base
    port_ft = args.port_ft
    base_model = args.model
    ft_model = os.getenv("FT_SERVED_MODEL_NAME", "ft")
    prompt = "Say hello in one short sentence."

    if args.mode in {"base", "both"}:
        _wait_ready(host, port_base, args.timeout_seconds)
        base_text = _post_chat(host, port_base, base_model, prompt)
        print(f"BASE: {base_text}")

    if args.mode in {"ft", "both"}:
        _wait_ready(host, port_ft, args.timeout_seconds)
        ft_text = _post_chat(host, port_ft, ft_model, prompt)
        print(f"FT: {ft_text}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        sys.exit(1)
