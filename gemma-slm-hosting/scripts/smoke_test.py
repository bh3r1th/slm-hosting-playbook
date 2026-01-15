import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _wait_ready(host: str, port: str) -> None:
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server not ready at {url} after 60s")


def _extract_text(payload: dict) -> str:
    if "text" in payload and isinstance(payload["text"], str):
        return payload["text"].strip()
    if "generated_text" in payload and isinstance(payload["generated_text"], str):
        return payload["generated_text"].strip()
    outputs = payload.get("outputs")
    if isinstance(outputs, list) and outputs:
        text = outputs[0].get("text")
        if isinstance(text, str):
            return text.strip()
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        message = choice.get("message", {})
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
        if isinstance(choice.get("text"), str):
            return choice["text"].strip()
    raise ValueError("Unable to extract text from response payload")


def _post_with_fallback(host: str, port: str, model_id: str, prompt: str) -> str:
    chat_url = f"http://{host}:{port}/v1/chat/completions"
    chat_payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 120,
    }
    response = requests.post(chat_url, json=chat_payload, timeout=30)
    if response.status_code not in {404, 405}:
        response.raise_for_status()
        return _extract_text(response.json())

    gen_url = f"http://{host}:{port}/inference/v1/generate"
    gen_payload = {"prompt": prompt, "max_new_tokens": 120, "temperature": 0.2}
    response = requests.post(gen_url, json=gen_payload, timeout=30)
    if response.ok:
        return _extract_text(response.json())

    body = response.text[:500]
    raise RuntimeError(
        f"All endpoints failed at {host}:{port}. Status={response.status_code}. Body={body}"
    )


def main() -> None:
    _load_env()
    host = os.getenv("VLLM_HOST", "127.0.0.1")
    port_base = os.getenv("VLLM_PORT_BASE", "8000")
    port_ft = os.getenv("VLLM_PORT_FT")
    base_model = os.getenv("BASE_MODEL_ID", "base")
    ft_model = os.getenv("ADAPTER_MODEL_NAME", "adapter")
    prompt = "Say hello in one short sentence."

    _wait_ready(host, port_base)
    base_text = _post_with_fallback(host, port_base, base_model, prompt)
    print(f"BASE: {base_text}")

    if port_ft:
        _wait_ready(host, port_ft)
        ft_text = _post_with_fallback(host, port_ft, ft_model, prompt)
        print(f"FT: {ft_text}")


if __name__ == "__main__":
    main()
