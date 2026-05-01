# Added by Codex: LoRA SFT trainer for answer-format discipline; not part of the original starter repository.

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from math_comp.data import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on answer-format SFT data.")
    parser.add_argument("--train", default="codex_added/data/sft_answer_format.jsonl", help="SFT JSONL path.")
    parser.add_argument("--output-dir", default="codex_added/models/qwen3_answer_format_lora", help="Adapter output directory.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507", help="Base model id.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--max-length", type=int, default=1536, help="Max tokenized sequence length.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--seed", type=int, default=151, help="Random seed.")
    parser.add_argument("--save-steps", type=int, default=100, help="Checkpoint save interval.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging interval.")
    parser.add_argument("--no-gradient-checkpointing", action="store_true", help="Disable gradient checkpointing.")
    return parser.parse_args()


class AnswerFormatDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.examples: list[dict[str, torch.Tensor]] = []
        self.skipped = 0
        for row in rows:
            messages = row["messages"]
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if len(full_ids) > max_length:
                self.skipped += 1
                continue
            labels = [-100] * len(full_ids)
            labels[len(prompt_ids) :] = full_ids[len(prompt_ids) :]
            self.examples.append(
                {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in features)
        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad_len = max_len - item["input_ids"].shape[0]
            batch["input_ids"].append(
                torch.cat([item["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            batch["attention_mask"].append(
                torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            batch["labels"].append(
                torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )
        return {key: torch.stack(value) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.train)
    random.shuffle(rows)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = AnswerFormatDataset(rows, tokenizer, args.max_length)
    if not dataset:
        raise RuntimeError("No training examples remain after max-length filtering.")
    print(f"Training examples: {len(dataset)}; skipped for length: {dataset.skipped}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    total_steps = math.ceil(len(dataset) / (args.batch_size * args.grad_accum) * args.epochs)
    print(f"Estimated optimizer steps: {total_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalCollator(pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
