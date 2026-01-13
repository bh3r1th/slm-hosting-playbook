import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from hf_auth import main as hf_login


def _format_examples(batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
    prompts = batch["prompt"]
    responses = batch["response"]
    texts = [f"User: {p}\nAssistant: {r}" for p, r in zip(prompts, responses)]
    return {"text": texts}


def _load_datasets(train_path: str, val_path: str):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train data: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing val data: {val_path}")

    dataset_dict = load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path},
    )
    dataset_dict = dataset_dict.map(_format_examples, batched=True)
    return dataset_dict["train"], dataset_dict["validation"]


def _build_peft_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def _build_model(model_id: str, qlora: bool):
    if qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
        )
        model = prepare_model_for_kbit_training(model)
        return model

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if torch.cuda.is_available():
        model.to("cuda")
    return model


def _setup_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _training_args(output_dir: str, epochs: int, lr: float, max_seq_len: int, run_name: str):
    report_to = ["wandb"] if os.getenv("WANDB_PROJECT") else []
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        report_to=report_to,
        run_name=run_name,
        max_steps=-1,
    )


def _write_training_config(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning for Gemma 3 1B IT.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    hf_login()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(root_dir, "data")
    train_path = os.path.join(data_dir, "processed_train.jsonl")
    val_path = os.path.join(data_dir, "processed_val.jsonl")

    train_ds, val_ds = _load_datasets(train_path, val_path)

    model_id = "google/gemma-3-1b-it"
    tokenizer = _setup_tokenizer(model_id)
    model = _build_model(model_id, args.qlora)
    peft_config = _build_peft_config()

    artifacts_dir = os.path.join(root_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    run_name = f"gemma-3-1b-it-lora-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    output_dir = os.path.join(artifacts_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        peft_config=peft_config,
        args=_training_args(output_dir, args.epochs, args.learning_rate, args.max_seq_len, run_name),
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    training_config_path = os.path.join(output_dir, "training_config.json")
    _write_training_config(
        training_config_path,
        {
            "model_id": model_id,
            "train_data": train_path,
            "val_data": val_path,
            "epochs": args.epochs,
            "max_seq_len": args.max_seq_len,
            "learning_rate": args.learning_rate,
            "qlora": args.qlora,
            "peft": "lora",
        },
    )


if __name__ == "__main__":
    main()
