import os

import requests


def _chat(host: str, port: str, model: str) -> str:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "temperature": 0.2,
        "max_tokens": 64,
    }
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def main() -> None:
    host = os.getenv("VLLM_HOST", "127.0.0.1")
    port_base = os.getenv("VLLM_PORT_BASE", "8000")
    port_ft = os.getenv("VLLM_PORT_FT", "8001")
    base_model = os.getenv("BASE_MODEL_ID", "base")
    ft_model = os.getenv("FT_MODEL_ID", "adapter")

    base_text = _chat(host, port_base, base_model)
    print(f"base response: {base_text}")

    ft_text = _chat(host, port_ft, ft_model)
    print(f"finetuned response: {ft_text}")


if __name__ == "__main__":
    main()
