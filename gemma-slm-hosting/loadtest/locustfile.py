"""Minimal Locust load test hitting the gateway."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

from locust import HttpUser, between, task

PROMPTS_PATH = Path(
    os.getenv("PROMPTS_FILE", "loadtest/workloads/prompts_short.jsonl")
)
ENDPOINT = os.getenv("ENDPOINT", "/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma")


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            prompt = payload.get("prompt")
            if prompt:
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


PROMPTS = load_prompts(PROMPTS_PATH)
PROMPTS_CYCLE = itertools.cycle(PROMPTS)


class GatewayUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def send_prompt(self) -> None:
        prompt = next(PROMPTS_CYCLE)
        if "chat/completions" in ENDPOINT:
            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
            }
        else:
            payload = {"prompt": prompt}
        self.client.post(
            ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            name="gateway_request",
        )
