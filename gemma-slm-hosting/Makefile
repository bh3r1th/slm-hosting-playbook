ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: help fmt lint test setup start-base start-ft start-both stop health smoke ab perf snapshot poc doctor env

SHELL := /bin/bash
MODE ?= both
PROMPTS ?= data/prompts.jsonl
OUT ?= runs/ab

help:
	@echo "gemma-slm-hosting"

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run pytest

env:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example"; else echo ".env already exists"; fi
	@echo "BASE_MODEL_ID=$${BASE_MODEL_ID:-}"
	@echo "ADAPTER_PATH=$${ADAPTER_PATH:-}"
	@echo "HOST=$${HOST:-}"
	@echo "BASE_PORT=$${BASE_PORT:-}"
	@echo "FT_PORT=$${FT_PORT:-}"
	@echo "BASE_API_URL=$${BASE_API_URL:-}"
	@echo "FT_API_URL=$${FT_API_URL:-}"

setup:
	mkdir -p runs/logs runs/pids runs/ab runs/perf runs/diag
	@if [ "$$SETUP_SKIP_GPU_CHECK" != "1" ]; then bash scripts/check_gpu.sh; else echo "Skipping GPU check"; fi
	python -m pip install -r requirements.txt
	python -c "import sys; print(sys.version.split()[0])"
	python -c "import torch; print(torch.__version__)"
	python -c "import vllm; print(vllm.__version__)"

start-base:
	bash scripts/start_base_vllm.sh

start-ft:
	bash scripts/start_ft_vllm.sh

start-both:
	bash scripts/start_base_vllm.sh
	bash scripts/start_ft_vllm.sh
	@echo "Started base and ft. PIDs in runs/pids/"

stop:
	bash scripts/stop_servers.sh

health:
	bash scripts/healthcheck.sh

smoke:
	python scripts/smoke_test.py --mode $(MODE)

ab:
	python eval/ab_eval.py --prompts $(PROMPTS) --out $(OUT)

perf:
	mkdir -p runs/perf
	@stamp=$$(date +%Y%m%d-%H%M%S); \
	python bench/perf.py --url "$$BASE_API_URL" --prompts $(PROMPTS) --out "runs/perf/base_$$stamp.json"; \
	python bench/perf.py --url "$$FT_API_URL" --prompts $(PROMPTS) --out "runs/perf/ft_$$stamp.json"

snapshot:
	bash scripts/diag_snapshot.sh

doctor:
	bash scripts/env_doctor.sh

poc:
	@set -e; \
	$(MAKE) stop || true; \
	$(MAKE) start-both; \
	$(MAKE) health; \
	$(MAKE) smoke MODE=both; \
	$(MAKE) ab PROMPTS=data/prompts.jsonl OUT=runs/ab; \
	if ! $(MAKE) perf; then echo "WARN: perf failed" >&2; fi; \
	$(MAKE) snapshot || true; \
	echo "Outputs: runs/logs runs/ab runs/perf runs/diag"
