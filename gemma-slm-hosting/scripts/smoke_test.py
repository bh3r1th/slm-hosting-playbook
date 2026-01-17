import argparse
import os
import sys
import time
from pathlib import Path
import requests
from dotenv import load_dotenv


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _wait_ready(api_url: str, timeout_seconds: int) -> None:
    url = f"{api_url}/v1/models"
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


def _post_chat(api_url: str, model_name: str, prompt: str) -> str:
    url = f"{api_url}/v1/chat/completions"
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


def _fetch_first_model(api_url: str, timeout_seconds: int) -> str:
    url = f"{api_url}/v1/models"
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    data = response.json()
    models = data.get("data", [])
    if not models:
        raise RuntimeError(f"No models returned from {url}")
    return models[0]["id"]


def _normalize_api_url(url: str) -> str:
    return url.rstrip("/")


def _fallback_api_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "ft", "both"], default="both")
    parser.add_argument(
        "--host", default=os.getenv("HOST", os.getenv("VLLM_HOST", "127.0.0.1"))
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=int(os.getenv("BASE_PORT", os.getenv("VLLM_PORT_BASE", "8000"))),
    )
    parser.add_argument(
        "--port-ft",
        type=int,
        default=int(os.getenv("FT_PORT", os.getenv("VLLM_PORT_FT", "8001"))),
    )
    parser.add_argument(
        "--model", default=os.getenv("BASE_MODEL_ID", "google/gemma-3-1b-it")
    )
    parser.add_argument("--base-api-url", default=os.getenv("BASE_API_URL", ""))
    parser.add_argument("--ft-api-url", default=os.getenv("FT_API_URL", ""))
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.getenv("TIMEOUT_SECONDS", "180")),
    )
    args = parser.parse_args()

    host = args.host
    port_base = args.port_base
    port_ft = args.port_ft
    ft_model = os.getenv("FT_SERVED_MODEL_NAME", "ft")
    prompt = "Say hello in one short sentence."

    base_api_url = _normalize_api_url(
        args.base_api_url or _fallback_api_url(host, port_base)
    )
    ft_api_url = _normalize_api_url(
        args.ft_api_url or _fallback_api_url(host, port_ft)
    )

    if args.model:
        base_model = args.model
        ft_model = args.model
    else:
        base_model = _fetch_first_model(base_api_url, args.timeout_seconds)
        if args.mode in {"ft", "both"}:
            ft_model = _fetch_first_model(ft_api_url, args.timeout_seconds)

    if args.mode in {"base", "both"}:
        _wait_ready(base_api_url, args.timeout_seconds)
        base_text = _post_chat(base_api_url, base_model, prompt)
        print(f"BASE: {base_text}")

    if args.mode in {"ft", "both"}:
        _wait_ready(ft_api_url, args.timeout_seconds)
        ft_text = _post_chat(ft_api_url, ft_model, prompt)
        print(f"FT: {ft_text}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        sys.exit(1)
