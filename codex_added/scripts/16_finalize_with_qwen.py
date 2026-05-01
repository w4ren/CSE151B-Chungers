# Added by Codex: second-stage Qwen final-answer formatter; not part of the original starter repository.

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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from judger import Judger
from math_comp.data import has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.prompts import format_options
from math_comp.scoring import extract_answer_key, score_item, summarize_results


SYSTEM = (
    "You are the same Qwen math model. Produce the final answer only. "
    "Use the problem and the previous Qwen reasoning trace. Do not call tools. "
    "For multiple-choice questions, output only one option letter inside \\boxed{}. "
    "For free-form questions, output the answer or ordered sub-answers inside one \\boxed{}. "
    "Do not include prose outside the box."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use Qwen to finalize raw traces into boxed answers.")
    parser.add_argument("--data", default="data/public.jsonl", help="Competition data JSONL.")
    parser.add_argument("--responses", required=True, help="JSONL with id and response fields.")
    parser.add_argument("--output", default="codex_added/results/finalized.jsonl", help="Finalized JSONL output.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507", help="Base model id.")
    parser.add_argument("--adapter-dir", default=None, help="Optional PEFT/LoRA adapter directory.")
    parser.add_argument("--only-missing", action="store_true", help="Only finalize rows without a strict answer key.")
    parser.add_argument("--max-trace-chars", type=int, default=6000, help="Keep the tail of long previous traces.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Finalizer generation budget.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Finalizer temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Finalizer top-p.")
    parser.add_argument("--score", action="store_true", help="Score output when data has answers.")
    return parser.parse_args()


def build_user_prompt(item: dict[str, Any], previous_response: str, max_trace_chars: int) -> str:
    question = str(item["question"])
    if item.get("options"):
        question += "\n\nOptions:\n" + format_options(item["options"])
    trace = previous_response[-max_trace_chars:]
    return (
        "Problem:\n"
        f"{question}\n\n"
        "Previous Qwen reasoning trace:\n"
        f"{trace}\n\n"
        "Final answer only:"
    )


def build_prompt(tokenizer: Any, item: dict[str, Any], previous_response: str, max_trace_chars: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_prompt(item, previous_response, max_trace_chars)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    forced_think = "<|im_start|>assistant\n<think>\n"
    if text.endswith(forced_think):
        text = text[: -len(forced_think)] + "<|im_start|>assistant\n"
    return text


def load_model(model_id: str, adapter_dir: str | None) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    data_by_id = index_by_id(data)
    rows = read_jsonl(args.responses)

    to_finalize: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for row in rows:
        item = data_by_id[int(row["id"])]
        strict_key = extract_answer_key(item, str(row.get("response", "")), strict=True)
        if args.only_missing and strict_key:
            kept.append({**row, "answer_key": strict_key})
        else:
            to_finalize.append(row)

    print(f"Finalizing {len(to_finalize)} rows; keeping {len(kept)} rows unchanged.")
    tokenizer, model = load_model(args.model_id, args.adapter_dir)

    finalized: list[dict[str, Any]] = kept[:]
    for row in tqdm(to_finalize, desc="Finalizing"):
        item = data_by_id[int(row["id"])]
        prompt = build_prompt(tokenizer, item, str(row.get("response", "")), args.max_trace_chars)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        record = {
            "id": int(row["id"]),
            "is_mcq": is_mcq(item),
            "source_response": str(row.get("response", "")),
            "response": response,
            "answer_key": extract_answer_key(item, response, strict=True),
        }
        for key in ("variant", "sample_index", "vote_count", "num_traces", "generated_tokens", "hit_token_limit"):
            if key in row:
                record[key] = row[key]
        finalized.append(record)

    finalized.sort(key=lambda record: int(record["id"]))
    if args.score:
        score_judger = Judger(strict_extract=False)
        for record in finalized:
            item = data_by_id[int(record["id"])]
            if has_gold(item):
                record["gold"] = item["answer"]
                record["correct"] = score_item(score_judger, item, str(record.get("response", "")))
        print(json.dumps(summarize_results(finalized), indent=2, sort_keys=True))
    write_jsonl(args.output, finalized)
    print(f"Wrote finalized responses to {args.output}")


if __name__ == "__main__":
    main()
