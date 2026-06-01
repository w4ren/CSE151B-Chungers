# Added by Codex: QLoRA solver SFT training for Qwen math reasoning.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer


SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step-by-step and put "
    "the final answer inside \\boxed{}."
)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML to use --config.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def bool_default(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    model_defaults = section(defaults, "model")
    train_defaults = section(defaults, "training")
    lora_defaults = section(defaults, "lora")
    path_defaults = section(defaults, "paths")

    parser = argparse.ArgumentParser(description="Train a separate QLoRA solver adapter with TRL SFTTrainer.")
    parser.add_argument("--config", default=None, help="Optional YAML config.")
    parser.add_argument("--model_name", "--model-name", default=model_defaults.get("name", "Qwen/Qwen3-4B-Thinking-2507"))
    parser.add_argument("--train_file", "--train-file", default=path_defaults.get("train_file", "codex_added/solver_sft/data/train.jsonl"))
    parser.add_argument("--eval_file", "--eval-file", default=path_defaults.get("eval_file", "codex_added/solver_sft/data/eval.jsonl"))
    parser.add_argument("--output_dir", "--output-dir", default=path_defaults.get("output_dir", "codex_added/solver_sft/outputs/qwen3_solver_sft_r32_8k"))
    parser.add_argument("--max_seq_length", "--max-seq-length", type=int, default=int(train_defaults.get("max_seq_length", 8192)))
    parser.add_argument("--per_device_train_batch_size", "--per-device-train-batch-size", type=int, default=int(train_defaults.get("per_device_train_batch_size", 1)))
    parser.add_argument("--per_device_eval_batch_size", "--per-device-eval-batch-size", type=int, default=int(train_defaults.get("per_device_eval_batch_size", 1)))
    parser.add_argument("--gradient_accumulation_steps", "--gradient-accumulation-steps", type=int, default=int(train_defaults.get("gradient_accumulation_steps", 8)))
    parser.add_argument("--num_train_epochs", "--num-train-epochs", type=float, default=float(train_defaults.get("num_train_epochs", 1.0)))
    parser.add_argument("--learning_rate", "--learning-rate", type=float, default=float(train_defaults.get("learning_rate", 1e-4)))
    parser.add_argument("--warmup_ratio", "--warmup-ratio", type=float, default=float(train_defaults.get("warmup_ratio", 0.03)))
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=float(train_defaults.get("weight_decay", 0.0)))
    parser.add_argument("--optim", default=train_defaults.get("optim", "paged_adamw_8bit"))
    parser.add_argument("--logging_steps", "--logging-steps", type=int, default=int(train_defaults.get("logging_steps", 10)))
    parser.add_argument("--save_steps", "--save-steps", type=int, default=int(train_defaults.get("save_steps", 500)))
    parser.add_argument("--eval_steps", "--eval-steps", type=int, default=train_defaults.get("eval_steps", None))
    parser.add_argument("--keep_last_n_checkpoints", "--keep-last-n-checkpoints", type=int, default=int(train_defaults.get("keep_last_n_checkpoints", 3)))
    parser.add_argument("--seed", type=int, default=int(train_defaults.get("seed", 151)))
    parser.add_argument("--dataset_num_proc", "--dataset-num-proc", type=int, default=train_defaults.get("dataset_num_proc", None))
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=bool_default(train_defaults.get("packing"), False))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=bool_default(train_defaults.get("bf16"), True))
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=bool_default(train_defaults.get("tf32"), True))
    parser.add_argument("--gradient_checkpointing", "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=bool_default(train_defaults.get("gradient_checkpointing"), True))
    parser.add_argument("--use_4bit", "--use-4bit", action=argparse.BooleanOptionalAction, default=bool_default(train_defaults.get("use_4bit"), True))
    parser.add_argument("--lora_rank", "--lora-rank", type=int, default=int(lora_defaults.get("rank", 32)))
    parser.add_argument("--lora_alpha", "--lora-alpha", type=int, default=int(lora_defaults.get("alpha", 64)))
    parser.add_argument("--lora_dropout", "--lora-dropout", type=float, default=float(lora_defaults.get("dropout", 0.05)))
    parser.add_argument("--resume_from_checkpoint", "--resume-from-checkpoint", default=train_defaults.get("resume_from_checkpoint", None))
    parser.add_argument("--report_to", "--report-to", nargs="*", default=train_defaults.get("report_to", []))
    parser.add_argument("--attn_implementation", "--attn-implementation", default=model_defaults.get("attn_implementation", None))
    return parser


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, remaining = pre.parse_known_args()
    config = load_yaml(known.config)
    parser = build_parser(config)
    return parser.parse_args(["--config", known.config] + remaining if known.config else remaining)


def answer_text(answer: Any) -> str:
    text = str(answer or "").strip()
    if text.startswith("\\boxed{") and text.endswith("}"):
        return text
    return f"\\boxed{{{text}}}"


def build_chat_text(tokenizer: Any, problem: str, reasoning: str, answer: str) -> str:
    assistant = f"<think>\n{reasoning.strip()}\n</think>\n\nThe answer is {answer_text(answer)}."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(problem).strip()},
        {"role": "assistant", "content": assistant},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def add_text_column(dataset: DatasetDict, tokenizer: Any, num_proc: int | None) -> DatasetDict:
    def convert_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        texts: list[str] = []
        for problem, reasoning, answer in zip(batch["problem"], batch["reasoning"], batch["answer"]):
            texts.append(build_chat_text(tokenizer, problem, reasoning, answer))
        return {"text": texts}

    remove_columns = list(dataset["train"].column_names)
    return dataset.map(
        convert_batch,
        batched=True,
        remove_columns=remove_columns,
        num_proc=num_proc,
        desc="Formatting chat SFT text",
    )


def local_device_map(use_4bit: bool) -> dict[str, int] | str | None:
    if not use_4bit:
        return None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        return {"": int(os.environ.get("LOCAL_RANK", "0"))}
    return "auto"


def load_model(args: argparse.Namespace) -> Any:
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = local_device_map(True)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def resolve_resume_checkpoint(output_dir: str, requested: str | None) -> str | None:
    if not requested:
        return None
    if requested.lower() == "last":
        checkpoint = get_last_checkpoint(output_dir)
        if not checkpoint:
            raise FileNotFoundError(f"No checkpoint found in {output_dir}")
        return checkpoint
    return requested


def main() -> None:
    args = parse_args()
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "eval": args.eval_file},
    )
    dataset = add_text_column(raw_dataset, tokenizer, args.dataset_num_proc)
    print(dataset)

    model = load_model(args)

    eval_strategy = "steps" if args.eval_steps else "no"
    sft_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim=args.optim,
        bf16=args.bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.keep_last_n_checkpoints,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        report_to=args.report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["eval"] if eval_strategy != "no" else None,
    }
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)

    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved solver LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
