# NOTE: vLLM requires a GPU runtime; use nvidia-container-runtime on a CUDA host.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv curl bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir vllm \
    && python3 -m pip install --no-cache-dir -r requirements.txt

ENV MODE=base

CMD ["bash", "-lc", "if [ \"$MODE\" = \"ft\" ]; then bash scripts/start_ft_vllm.sh; else bash scripts/start_base_vllm.sh; fi"]
