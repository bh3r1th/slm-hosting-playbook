import argparse
import os
from typing import Optional

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_auth import main as hf_login


def _latest_adapter_dir(artifacts_dir: str) -> Optional[str]:
    candidates = []
    for name in os.listdir(artifacts_dir):
        path = os.path.join(artifacts_dir, name)
        if not os.path.isdir(path):
            continue
        if name == "merged":
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def _validate_adapter(adapter_dir: str) -> None:
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")


def _write_pointer(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.strip() + "\n")


def _merge_and_save(base_model_id: str, adapter_dir: str, merged_dir: str) -> str:
    hf_login()
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged = model.merge_and_unload()
    os.makedirs(merged_dir, exist_ok=True)
    merged.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    tokenizer.save_pretrained(merged_dir)
    return merged_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LoRA artifacts for Phase 2 hosting.")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--base-model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    artifacts_dir = os.path.join(root_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    adapter_dir = args.adapter_path or _latest_adapter_dir(artifacts_dir)
    if not adapter_dir:
        raise FileNotFoundError("No adapter directory found in artifacts.")
    _validate_adapter(adapter_dir)

    pointer_path = os.path.join(artifacts_dir, "PHASE2_MODEL_POINTER.txt")

    if args.merge:
        merged_dir = os.path.join(artifacts_dir, "merged")
        merged_path = _merge_and_save(args.base_model, adapter_dir, merged_dir)
        _write_pointer(pointer_path, f"MERGED_MODEL_PATH={merged_path}")
        print(f"Merged model saved: {merged_path}")
        print(f"Phase 2 pointer: {pointer_path}")
        return

    print(f"Adapter path: {adapter_dir}")
    print(f"Base model: {args.base_model}")
    _write_pointer(
        pointer_path,
        f"ADAPTER_PATH={adapter_dir}\nBASE_MODEL_ID={args.base_model}",
    )
    print(f"Phase 2 pointer: {pointer_path}")


if __name__ == "__main__":
    main()
