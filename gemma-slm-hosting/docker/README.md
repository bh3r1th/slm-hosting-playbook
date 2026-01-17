This folder is illustrative; Colab does not run Docker GPU workloads.
Use on a CUDA-capable host with NVIDIA Container Toolkit.

Example:
docker compose -f docker/docker-compose.yml up --build

Set BASE_MODEL_ID and provide adapters at docker/adapters/.
