# Added by Codex: GRPO training for a category-routed answer finalizer LoRA.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from math_comp.data import read_jsonl
from math_comp.grpo_finalizer import (
    FinalizerRewardWeights,
    completion_text,
    direct_chat_prompt,
    score_finalizer_completion,
)


def default_model_id() -> str:
    snapshot_root = REPO_ROOT / ".hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots"
    snapshots = [path for path in snapshot_root.glob("*") if path.is_dir()]
    if snapshots:
        return str(max(snapshots, key=lambda path: path.stat().st_mtime))
    return "Qwen/Qwen3-4B-Thinking-2507"


def default_init_adapter() -> str:
    rebuilt = REPO_ROOT / "codex_added/models/qwen3_answer_format_lora_rebuilt_20260531_070427"
    if rebuilt.exists():
        return str(rebuilt)
    return "codex_added/models/qwen3_answer_format_lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRPO finalizer adapter from public trace-extraction data.")
    parser.add_argument("--train-file", default="codex_added/data/grpo_finalizer/train.jsonl")
    parser.add_argument("--output-dir", default="codex_added/models/qwen3_grpo_finalizer")
    parser.add_argument("--model-id", default=default_model_id())
    parser.add_argument("--init-adapter-dir", default=default_init_adapter())
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1520)
    parser.add_argument("--reward-log", default="codex_added/results/grpo_finalizer_reward_log.jsonl")
    parser.add_argument("--reward-log-limit", type=int, default=500)
    parser.add_argument("--smoke", action="store_true", help="Use 32 examples, 30 steps, 4 generations, and 64 tokens.")
    return parser.parse_args()


def load_tokenizer(model_id: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_dataset(path: str, tokenizer: Any, max_examples: int | None) -> Dataset:
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[:max_examples]
    examples: list[dict[str, Any]] = []
    for row in rows:
        examples.append(
            {
                "prompt": direct_chat_prompt(tokenizer, row["messages"]),
                "id": int(row["id"]),
                "question": str(row["problem"]),
                "options_json": json.dumps(row.get("options") or [], ensure_ascii=False),
                "answer_json": json.dumps(row["answer"], ensure_ascii=False),
                "is_mcq": bool(row.get("format_profile", {}).get("is_mcq")),
                "num_ans_slots": int(row.get("format_profile", {}).get("num_ans_slots") or 1),
                "expected_answer_kind": str(row.get("format_profile", {}).get("expected_answer_kind") or "unknown"),
                "raw_extracted_answer": str(row.get("raw_extracted_answer") or ""),
                "raw_correct": row.get("raw_correct"),
                "current_finalizer_correct": row.get("current_finalizer_correct"),
            }
        )
    if not examples:
        raise RuntimeError(f"No GRPO finalizer examples found in {path}")
    return Dataset.from_list(examples)


def load_model(args: argparse.Namespace) -> tuple[Any, LoraConfig | None]:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.init_adapter_dir:
        return PeftModel.from_pretrained(model, args.init_adapter_dir, is_trainable=True), None
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return model, peft_config


def make_reward_func(log_path: str | None, log_limit: int) -> Any:
    weights = FinalizerRewardWeights()
    count = 0
    out_path = Path(log_path) if log_path else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")

    def reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
        nonlocal count
        rewards: list[float] = []
        ids = kwargs.get("id", [])
        questions = kwargs.get("question", [])
        options_list = kwargs.get("options_json", [])
        answers = kwargs.get("answer_json", [])
        raw_keys = kwargs.get("raw_extracted_answer", [])
        kinds = kwargs.get("expected_answer_kind", [])
        slots = kwargs.get("num_ans_slots", [])
        for index, completion in enumerate(completions):
            item = {
                "id": int(ids[index]) if index < len(ids) else index,
                "question": str(questions[index]) if index < len(questions) else "",
                "options": json.loads(options_list[index]) if index < len(options_list) else [],
                "answer": json.loads(answers[index]) if index < len(answers) else "",
            }
            result = score_finalizer_completion(
                item,
                completion,
                raw_extracted_answer=str(raw_keys[index]) if index < len(raw_keys) else "",
                weights=weights,
            )
            rewards.append(float(result["reward"]))
            if out_path and count < log_limit:
                with out_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "id": item["id"],
                                "expected_answer_kind": str(kinds[index]) if index < len(kinds) else "",
                                "num_ans_slots": int(slots[index]) if index < len(slots) else None,
                                "completion": completion_text(completion),
                                **result,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                count += 1
        return rewards

    return reward_func


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_examples = 32
        args.max_steps = min(args.max_steps, 30)
        args.num_generations = min(args.num_generations, 4)
        args.max_completion_length = min(args.max_completion_length, 64)

    tokenizer = load_tokenizer(args.model_id)
    dataset = build_dataset(args.train_file, tokenizer, args.max_examples)
    model, peft_config = load_model(args)

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=0.7,
        top_p=0.95,
        top_k=20,
        beta=0.0,
        use_vllm=False,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to=[],
        remove_unused_columns=False,
        save_safetensors=True,
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_func(args.reward_log, args.reward_log_limit),
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved GRPO finalizer adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
