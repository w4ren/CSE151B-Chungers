# Added by Codex: generic chat-message LoRA/QLoRA SFT trainer for selector/finalizer/solver datasets.

from __future__ import annotations

import argparse
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
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from math_comp.data import read_jsonl


def default_model_id() -> str:
    snapshot_root = REPO_ROOT / ".hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots"
    snapshots = [path for path in snapshot_root.glob("*") if path.is_dir()]
    if snapshots:
        return str(max(snapshots, key=lambda path: path.stat().st_mtime))
    return "Qwen/Qwen3-4B-Thinking-2507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter from chat-message JSONL rows.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=default_model_id())
    parser.add_argument("--init-adapter-dir", default=None, help="Optional adapter to continue from.")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1524)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class ChatDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.examples: list[dict[str, torch.Tensor]] = []
        self.skipped = 0
        for row in rows:
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                self.skipped += 1
                continue
            prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in features)
        out: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_len - item["input_ids"].shape[0]
            out["input_ids"].append(torch.cat([item["input_ids"], torch.full((pad,), self.pad_token_id, dtype=torch.long)]))
            out["attention_mask"].append(torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            out["labels"].append(torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        return {key: torch.stack(value) for key, value in out.items()}


def load_rows(path: str, max_examples: int | None, seed: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    random.Random(seed).shuffle(rows)
    if max_examples is not None:
        rows = rows[:max_examples]
    return rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = ChatDataset(load_rows(args.train, args.max_examples, args.seed), tokenizer, args.max_length)
    eval_dataset = ChatDataset(read_jsonl(args.eval), tokenizer, args.max_length) if args.eval else None
    if len(train_dataset) == 0:
        raise RuntimeError(f"No train examples remain after max-length filtering; skipped={train_dataset.skipped}")
    print(f"Train examples: {len(train_dataset)}; skipped: {train_dataset.skipped}")
    if eval_dataset:
        print(f"Eval examples: {len(eval_dataset)}; skipped: {eval_dataset.skipped}")

    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": torch.bfloat16, "device_map": "auto"}
    if args.use_4bit:
        model_kwargs["load_in_4bit"] = True
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.init_adapter_dir:
        model = PeftModel.from_pretrained(model, args.init_adapter_dir, is_trainable=True)
    else:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, config)
    model.print_trainable_parameters()

    estimated_steps = math.ceil(len(train_dataset) / (args.batch_size * args.grad_accum) * args.epochs)
    if args.max_steps > 0:
        estimated_steps = min(estimated_steps, args.max_steps)
    print(f"Estimated optimizer steps: {estimated_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=4,
        eval_strategy="steps" if eval_dataset and args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
