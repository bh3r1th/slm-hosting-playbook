import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, load_dataset

from hf_auth import main as hf_login


def _find_lang_field(columns: Iterable[str]) -> Optional[str]:
    for name in ("lang", "language", "locale"):
        if name in columns:
            return name
    return None


def _filter_english(ds: Dataset) -> Dataset:
    lang_field = _find_lang_field(ds.column_names)
    if not lang_field:
        return ds
    return ds.filter(lambda row: row.get(lang_field) == "en")


def _is_user_role(role: Optional[str]) -> bool:
    if not role:
        return False
    return role.lower() in {"user", "prompter", "human"}


def _is_assistant_role(role: Optional[str]) -> bool:
    if not role:
        return False
    return role.lower() in {"assistant", "bot"}


def _build_pairs(ds: Dataset) -> List[Dict[str, str]]:
    by_id: Dict[str, Dict[str, str]] = {}
    for row in ds:
        message_id = row.get("message_id")
        if message_id:
            by_id[message_id] = row

    pairs: List[Dict[str, str]] = []
    for row in ds:
        if not _is_assistant_role(row.get("role")):
            continue
        parent_id = row.get("parent_id")
        if not parent_id:
            continue
        parent = by_id.get(parent_id)
        if not parent or not _is_user_role(parent.get("role")):
            continue
        prompt = parent.get("text")
        completion = row.get("text")
        if not prompt or not completion:
            continue
        pairs.append({"prompt": prompt, "completion": completion})
    return pairs


def _cap(pairs: List[Dict[str, str]], max_items: Optional[int]) -> List[Dict[str, str]]:
    if not max_items:
        return pairs
    return pairs[:max_items]


def _write_jsonl(path: str, rows: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_split(ds: Dataset, max_items: Optional[int]) -> List[Dict[str, str]]:
    filtered = _filter_english(ds)
    pairs = _build_pairs(filtered)
    return _cap(pairs, max_items)


def _resolve_splits(dataset_dict) -> Tuple[Dataset, Dataset]:
    if "validation" in dataset_dict:
        return dataset_dict["train"], dataset_dict["validation"]
    if "train" in dataset_dict:
        shuffled = dataset_dict["train"].shuffle(seed=42)
        val_size = min(2000, max(1, int(len(shuffled) * 0.02)))
        return shuffled.select(range(len(shuffled) - val_size)), shuffled.select(
            range(len(shuffled) - val_size, len(shuffled))
        )
    raise ValueError("Dataset has no train or validation split.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OASST1 SFT data.")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    args = parser.parse_args()

    hf_login()
    dataset_dict = load_dataset("OpenAssistant/oasst1")
    train_ds, val_ds = _resolve_splits(dataset_dict)

    train_pairs = _prepare_split(train_ds, args.max_train)
    val_pairs = _prepare_split(val_ds, args.max_val)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "processed_train.jsonl")
    val_path = os.path.join(output_dir, "processed_val.jsonl")
    _write_jsonl(train_path, train_pairs)
    _write_jsonl(val_path, val_pairs)

    print(f"Train examples: {len(train_pairs)}")
    print(f"Val examples: {len(val_pairs)}")
    sample = train_pairs[0] if train_pairs else (val_pairs[0] if val_pairs else None)
    if sample:
        print("Sample:")
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
