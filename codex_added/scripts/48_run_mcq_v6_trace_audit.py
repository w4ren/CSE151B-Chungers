# Added by Codex: constrained MCQ auditor that uses the current V6 trace as context.

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm
from transformers import AutoTokenizer

from math_comp.data import batched, index_by_id, is_mcq, read_jsonl
from math_comp.prompts import format_options
from math_comp.scoring import extract_answer_key
from math_comp.vllm_helpers import add_vllm_cli_args, make_llm, make_sampling_params, vllm_generated_tokens, vllm_text

AUDIT_VARIANTS: dict[str, dict[str, str]] = {
    "trace_audit": {
        "system": "You are a careful multiple-choice math answer auditor. Choose the correct option letter only.",
        "task": (
            "Audit the previous solution. It may be correct or wrong. Use the problem, options, and trace to choose "
            "the mathematically correct option. Output exactly one option letter and no explanation."
        ),
    },
    "trace_skeptical": {
        "system": "You are a skeptical math contest grader. Detect option-letter and option-text mismatches.",
        "task": (
            "The previous answer may contain a subtle mistake or may choose an option whose written form does not "
            "match the derived result. Recheck the work and output exactly one option letter."
        ),
    },
    "trace_final_match": {
        "system": "You map solved mathematical results to multiple-choice options exactly.",
        "task": (
            "Focus on the final derived mathematical expression in the trace, not just the previously chosen letter. "
            "Select the option whose text most literally matches the correct derived result. Output one option letter."
        ),
    },
    "trace_independent": {
        "system": "You are an independent multiple-choice math solver and verifier.",
        "task": (
            "Use the previous trace only as scratch evidence. Independently decide which listed option is correct. "
            "Output exactly one option letter and no other text."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run constrained one-token MCQ audits using V6 traces as context.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", default=",".join(AUDIT_VARIANTS))
    parser.add_argument("--trace-head-chars", type=int, default=1800)
    parser.add_argument("--trace-tail-chars", type=int, default=4200)
    parser.add_argument("--model-id", required=True)
    add_vllm_cli_args(parser, default_backend="vllm")
    return parser.parse_args()


def compact_trace(text: str, *, head_chars: int, tail_chars: int) -> str:
    text = str(text or "").strip()
    budget = max(0, head_chars) + max(0, tail_chars)
    if not text or len(text) <= budget:
        return text
    return text[:head_chars].rstrip() + "\n\n... [middle of V6 trace omitted] ...\n\n" + text[-tail_chars:].lstrip()


def direct_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    forced = "<|im_start|>assistant\n<think>\n"
    if text.endswith(forced):
        text = text[: -len(forced)] + "<|im_start|>assistant\n"
    return text


def allowed_letter_token_ids(tokenizer: Any, option_count: int) -> list[int]:
    ids: set[int] = set()
    for index in range(option_count):
        label = chr(65 + index)
        for text in (label, " " + label):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(int(encoded[0]))
    if not ids:
        raise ValueError("Could not build one-token option-letter constraint")
    return sorted(ids)


def build_prompt(
    tokenizer: Any,
    item: dict[str, Any],
    baseline_row: dict[str, Any],
    variant_name: str,
    *,
    trace_head_chars: int,
    trace_tail_chars: int,
) -> str:
    variant = AUDIT_VARIANTS[variant_name]
    baseline_final_response = str(baseline_row.get("response", ""))
    baseline_trace = str(baseline_row.get("source_response") or baseline_final_response)
    baseline_key = extract_answer_key(item, baseline_final_response, strict=True) or "UNKNOWN"
    trace = compact_trace(baseline_trace, head_chars=trace_head_chars, tail_chars=trace_tail_chars)
    letters = ", ".join(chr(65 + idx) for idx, _ in enumerate(item.get("options") or []))
    user = (
        f"Problem:\n{item['question']}\n\n"
        f"Options:\n{format_options(item.get('options') or [])}\n\n"
        f"Previous V6 extracted option: {baseline_key}\n\n"
        f"Previous V6 solution trace excerpt:\n{trace}\n\n"
        f"Task:\n{variant['task']}\n\n"
        f"Allowed output letters: {letters}. Output exactly one letter."
    )
    messages = [{"role": "system", "content": variant["system"]}, {"role": "user", "content": user}]
    return direct_chat_prompt(tokenizer, messages)


def main() -> None:
    args = parse_args()
    if args.backend != "vllm":
        raise ValueError("This audit runner currently supports --backend vllm only.")

    data = [row for row in read_jsonl(args.data) if is_mcq(row)]
    if args.offset:
        data = data[args.offset :]
    if args.limit is not None:
        data = data[: args.limit]
    baseline_by_id = index_by_id(read_jsonl(args.baseline))
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in variant_names if name not in AUDIT_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown audit variants: {unknown}; available={sorted(AUDIT_VARIANTS)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    llm = make_llm(args.model_id, args, trust_remote_code=True)

    tasks: list[dict[str, Any]] = []
    for item in data:
        if int(item["id"]) not in baseline_by_id:
            raise KeyError(f"Missing baseline row for id {item['id']}")
        for sample_index, variant_name in enumerate(variant_names):
            tasks.append({"item": item, "variant": variant_name, "sample_index": sample_index})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, int(args.vllm_batch_size or 1))
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in tqdm(list(batched(tasks, chunk_size)), desc="MCQ V6 audit"):
            prompts = [
                build_prompt(
                    tokenizer,
                    task["item"],
                    baseline_by_id[int(task["item"]["id"])],
                    task["variant"],
                    trace_head_chars=args.trace_head_chars,
                    trace_tail_chars=args.trace_tail_chars,
                )
                for task in chunk
            ]
            sampling_params = [
                make_sampling_params(
                    max_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=None,
                    do_sample=False,
                    allowed_token_ids=allowed_letter_token_ids(tokenizer, len(task["item"].get("options") or [])),
                )
                for task in chunk
            ]
            outputs = llm.generate(prompts, sampling_params=sampling_params)
            for task, output in zip(chunk, outputs):
                item = task["item"]
                raw_text = vllm_text(output).strip()
                answer_key = extract_answer_key(item, raw_text, strict=True)
                if not answer_key:
                    answer_key = raw_text.strip().upper()[:1]
                record = {
                    "id": int(item["id"]),
                    "is_mcq": True,
                    "variant": "mcq_v6_trace_audit",
                    "audit_variant": task["variant"],
                    "sample_index": int(task["sample_index"]),
                    "answer_key": answer_key,
                    "generated_tokens": vllm_generated_tokens(output),
                    "hit_token_limit": False,
                    "response": f"\\boxed{{{answer_key}}}",
                    "raw_response": raw_text,
                    "baseline_answer_key": extract_answer_key(item, str(baseline_by_id[int(item["id"])].get("response", "")), strict=True),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    print(f"Wrote {len(tasks)} constrained MCQ audit rows to {output_path}")


if __name__ == "__main__":
    main()
