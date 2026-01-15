import os
from pathlib import Path


def _read_kv(path: Path) -> dict:
    data = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent.parent
    training_dir = base_dir / "gemma-slm-training"
    pointer = training_dir / "PHASE2_MODEL_POINTER.txt"
    fallback = training_dir / "PHASE2_MODEL_POINTER.example.txt"

    if pointer.exists():
        values = _read_kv(pointer)
    else:
        values = _read_kv(fallback)

    base_model_id = os.getenv("BASE_MODEL_ID", values.get("BASE_MODEL_ID", ""))
    adapter_path = os.getenv("ADAPTER_PATH", values.get("ADAPTER_PATH", ""))

    print(f"export BASE_MODEL_ID={base_model_id}")
    print(f"export ADAPTER_PATH={adapter_path}")


if __name__ == "__main__":
    main()
