# Added by Codex: public-answer GRPO trainer; not part of the original starter repository.

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

from judger import Judger
from math_comp.data import answer_slot_count, has_gold, is_mcq, read_jsonl
from math_comp.prompts import format_options
from math_comp.scoring import extract_answer_key, score_item
from math_comp.variants import get_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Qwen LoRA adapter with public-answer GRPO rewards.")
    parser.add_argument("--public", default="data/public.jsonl", help="Public JSONL with answers.")
    parser.add_argument("--output-dir", default="codex_added/models/qwen3_grpo_public_lora", help="Output adapter dir.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507", help="Required base model.")
    parser.add_argument("--init-adapter-dir", default=None, help="Optional trainable PEFT adapter to continue from.")
    parser.add_argument("--variant", default="final_box_only", help="Prompt variant used for GRPO prompts.")
    parser.add_argument(
        "--assistant-mode",
        choices=["think", "direct"],
        default="think",
        help="Use Qwen's thinking chat template or suppress the forced thinking prefill for shorter RL completions.",
    )
    parser.add_argument("--max-examples", type=int, default=64, help="Limit public examples for a first run.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N public examples.")
    parser.add_argument(
        "--exclude-ids-file",
        default=None,
        help="Optional JSON/list/text file of public ids to exclude before offset and max_examples.",
    )
    parser.add_argument("--max-steps", type=int, default=20, help="GRPO optimizer steps.")
    parser.add_argument("--num-generations", type=int, default=2, help="Completions per prompt group.")
    parser.add_argument("--max-prompt-length", type=int, default=1024, help="Token budget for prompts.")
    parser.add_argument("--max-completion-length", type=int, default=512, help="Token budget for completions.")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="LoRA learning rate.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank when not continuing an adapter.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha when not continuing an adapter.")
    parser.add_argument("--temperature", type=float, default=0.6, help="GRPO sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="GRPO top-p.")
    parser.add_argument("--top-k", type=int, default=20, help="GRPO top-k.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N optimizer steps.")
    parser.add_argument("--save-steps", type=int, default=0, help="Optional checkpoint interval; 0 disables step saves.")
    parser.add_argument("--debug-reward-log", default=None, help="Optional JSONL file with sampled reward inputs.")
    parser.add_argument("--debug-reward-limit", type=int, default=12, help="Max rows written to --debug-reward-log.")
    parser.add_argument("--correct-reward", type=float, default=1.0, help="Reward for exact correctness.")
    parser.add_argument("--boxed-reward", type=float, default=0.0, help="Optional reward for any strict boxed answer.")
    parser.add_argument("--mcq-letter-reward", type=float, default=0.0, help="Optional reward for one-letter MCQ format.")
    parser.add_argument("--slot-count-reward", type=float, default=0.0, help="Optional reward for free-form slot count.")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="Use bf16 training.")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true", help="Use tiny settings: 4 examples and 2 GRPO steps.")
    return parser.parse_args()


def read_exclude_ids(path: str | None) -> set[int]:
    if not path:
        return set()
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return set()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "ordered_selected_ids" in data:
                return {int(x) for x in data["ordered_selected_ids"]}
            if "ids" in data:
                return {int(x) for x in data["ids"]}
        if isinstance(data, list):
            return {int(x) for x in data}
    except json.JSONDecodeError:
        pass

    return {int(line.strip()) for line in text.splitlines() if line.strip()}


def build_problem_text(item: dict[str, Any]) -> str:
    question = str(item["question"])
    if item.get("options"):
        question += "\n\nOptions:\n" + format_options(item["options"])
    return question


def direct_prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    forced_think = "<|im_start|>assistant\n<think>\n"
    if text.endswith(forced_think):
        text = text[: -len(forced_think)] + "<|im_start|>assistant\n"
    return text


def build_dataset(
    path: str,
    variant_name: str,
    offset: int,
    max_examples: int | None,
    tokenizer: Any,
    assistant_mode: str,
    exclude_ids: set[int] | None = None,
) -> Dataset:
    variant = get_variant(variant_name)
    rows: list[dict[str, Any]] = []
    items = [item for item in read_jsonl(path) if has_gold(item)]
    if exclude_ids:
        items = [item for item in items if int(item["id"]) not in exclude_ids]
    if offset:
        items = items[offset:]
    if max_examples is not None:
        items = items[:max_examples]

    for item in items:
        system = variant["mcq_system"] if is_mcq(item) else variant["math_system"]
        slot_note = (
            "Choose exactly one option letter."
            if is_mcq(item)
            else f"There are {answer_slot_count(item)} [ANS] slot(s). Put exactly that many ordered sub-answer(s) in one box."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{build_problem_text(item)}\n\n{slot_note}"},
        ]
        prompt: Any = messages if assistant_mode == "think" else direct_prompt_text(tokenizer, messages)
        rows.append(
            {
                "prompt": prompt,
                "id": int(item["id"]),
                "question": str(item["question"]),
                "options_json": json.dumps(item.get("options") or [], ensure_ascii=False),
                "answer_json": json.dumps(item["answer"], ensure_ascii=False),
                "is_mcq": is_mcq(item),
            }
        )

    if not rows:
        raise ValueError(f"No answered public examples found in {path}")
    return Dataset.from_list(rows)


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(completion)


def make_reward_func(
    debug_reward_log: str | None = None,
    debug_reward_limit: int = 12,
    correct_reward: float = 1.0,
    boxed_reward: float = 0.0,
    mcq_letter_reward: float = 0.0,
    slot_count_reward: float = 0.0,
) -> Any:
    score_judger = Judger(strict_extract=False)
    strict_judger = Judger(strict_extract=True)
    debug_count = 0
    debug_path = Path(debug_reward_log) if debug_reward_log else None
    if debug_path:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text("", encoding="utf-8")

    def reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
        nonlocal debug_count
        rewards: list[float] = []
        answers = kwargs.get("answer_json", [])
        questions = kwargs.get("question", [])
        options_list = kwargs.get("options_json", [])
        is_mcq_list = kwargs.get("is_mcq", [])

        for completion, answer_json, question, options_json, item_is_mcq in zip(
            completions,
            answers,
            questions,
            options_list,
            is_mcq_list,
        ):
            response = completion_text(completion)
            answer = json.loads(answer_json)
            options = json.loads(options_json)
            item = {
                "question": question,
                "options": options or [],
                "answer": answer,
            }
            try:
                correct = score_item(score_judger, item, response)
            except Exception:
                correct = False

            key = extract_answer_key(item, response, strict=True)
            reward = correct_reward if correct else 0.0
            if key or "\\boxed{" in response:
                reward += boxed_reward
            if item_is_mcq and key and len(str(key).strip()) == 1:
                reward += mcq_letter_reward
            if not item_is_mcq and key:
                expected_slots = len(answer) if isinstance(answer, list) else 1
                pred_slots = len(strict_judger.split_by_comma(strict_judger.extract_ans(response) or ""))
                if pred_slots == expected_slots:
                    reward += slot_count_reward
            rewards.append(reward)
            if debug_path and debug_count < debug_reward_limit:
                with debug_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "question": str(question)[:500],
                                "answer": answer,
                                "is_mcq": bool(item_is_mcq),
                                "response": response,
                                "answer_key": key,
                                "correct": bool(correct),
                                "reward": reward,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                debug_count += 1
        return rewards

    return reward_func


def load_tokenizer(model_id: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_trainable_model(model_id: str, init_adapter_dir: str | None) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if init_adapter_dir:
        model = PeftModel.from_pretrained(model, init_adapter_dir, is_trainable=True)
    return model


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_examples = 4
        args.max_steps = 2
        args.max_completion_length = min(args.max_completion_length, 256)

    tokenizer = load_tokenizer(args.model_id)
    exclude_ids = read_exclude_ids(args.exclude_ids_file)
    dataset = build_dataset(
        args.public,
        args.variant,
        args.offset,
        args.max_examples,
        tokenizer,
        args.assistant_mode,
        exclude_ids,
    )
    model = load_trainable_model(args.model_id, args.init_adapter_dir)

    peft_config = None
    if not args.init_adapter_dir:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    save_strategy = "no" if args.save_steps <= 0 else "steps"
    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        beta=0.0,
        use_vllm=False,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=max(args.save_steps, 1),
        report_to=[],
        remove_unused_columns=False,
        save_safetensors=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_func(
            args.debug_reward_log,
            args.debug_reward_limit,
            args.correct_reward,
            args.boxed_reward,
            args.mcq_letter_reward,
            args.slot_count_reward,
        ),
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "model_id": args.model_id,
        "public": args.public,
        "variant": args.variant,
        "max_examples": len(dataset),
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "init_adapter_dir": args.init_adapter_dir,
        "exclude_ids_file": args.exclude_ids_file,
        "excluded_id_count": len(exclude_ids),
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "codex_grpo_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved GRPO adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
