# Added by Codex: second-stage Qwen final-answer formatter; not part of the original starter repository.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from judger import Judger
from math_comp.data import (
    append_jsonl_record,
    answer_slot_count,
    batched,
    has_gold,
    index_by_id,
    is_mcq,
    read_jsonl,
    read_jsonl_complete_prefix,
    write_jsonl,
)
from math_comp.final_answer import final_answer_diagnostics, final_format_contract, normalize_final_response
from math_comp.prompts import format_options
from math_comp.scoring import extract_answer_key, score_item, summarize_results
from math_comp.vllm_helpers import add_vllm_cli_args, make_llm, make_lora_request, make_sampling_params, vllm_text


SYSTEM = (
    "You are the same Qwen math model. Produce the final answer only. "
    "Use the problem and the previous Qwen reasoning trace. Do not call tools. "
    "Extract and normalize the final answer; do not re-solve unless the trace is unusable. "
    "Obey the requested output schema exactly. Never summarize multiple slots into one value. "
    "Do not include prose outside the final box."
)

RecordSink = Callable[[dict[str, Any]], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use Qwen to finalize raw traces into boxed answers.")
    parser.add_argument("--data", default="data/public.jsonl", help="Competition data JSONL.")
    parser.add_argument("--responses", required=True, help="JSONL with id and response fields.")
    parser.add_argument("--output", default="codex_added/results/finalized.jsonl", help="Finalized JSONL output.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507", help="Base model id.")
    parser.add_argument("--adapter-dir", default=None, help="Optional PEFT/LoRA adapter directory.")
    parser.add_argument("--only-missing", action="store_true", help="Only finalize rows without a strict answer key.")
    parser.add_argument("--max-trace-chars", type=int, default=6000, help="Keep the tail of long previous traces.")
    parser.add_argument("--max-input-tokens", type=int, default=4096, help="Tokenizer truncation budget.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Finalizer generation budget.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Finalizer temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Finalizer top-p.")
    parser.add_argument(
        "--retry-bad-format",
        action="store_true",
        help="Retry free-form rows whose finalized answer has the wrong number of slots.",
    )
    parser.add_argument(
        "--retry-max-new-tokens",
        type=int,
        default=128,
        help="Generation budget for the stricter bad-format retry pass.",
    )
    parser.add_argument("--score", action="store_true", help="Score output when data has answers.")
    add_vllm_cli_args(parser)
    return parser.parse_args()


def extraction_rules(item: dict[str, Any], retry_reason: str | None = None) -> str:
    rules: list[str] = []
    if retry_reason:
        rules.append(f"- The previous finalizer attempt was rejected because {retry_reason}.")
    if is_mcq(item):
        rules.extend(
            [
                "- Choose exactly one option letter from the listed choices.",
                "- Match the option text and notation requested by the problem, not only an equivalent alternate form.",
            ]
        )
    else:
        slots = answer_slot_count(item)
        rules.extend(
            [
                f"- The problem has {slots} [ANS] slot(s); output exactly {slots} comma-separated field(s).",
                "- Fill the slots in the same order they appear in the problem.",
                "- If the trace contains a table/list of intermediate requested values, include every requested value.",
                "- Do not keep only the final statistic when earlier table cells are also [ANS] slots.",
                "- Preserve exact expressions and the most precise decimals present in the trace unless the problem explicitly requests rounding.",
            ]
        )
    return "\n".join(rules)


def build_user_prompt(
    item: dict[str, Any],
    previous_response: str,
    max_trace_chars: int,
    retry_reason: str | None = None,
    first_attempt: str | None = None,
) -> str:
    question = str(item["question"])
    if item.get("options"):
        question += "\n\nOptions:\n" + format_options(item["options"])
    trace = previous_response[-max_trace_chars:]
    first_attempt_block = f"Rejected first finalizer attempt:\n{first_attempt}\n\n" if first_attempt else ""
    return (
        "Problem:\n"
        f"{question}\n\n"
        "Required final-answer schema:\n"
        f"{final_format_contract(item)}\n\n"
        "Extraction rules:\n"
        f"{extraction_rules(item, retry_reason)}\n\n"
        f"{first_attempt_block}"
        "Previous Qwen reasoning trace:\n"
        f"{trace}\n\n"
        "Final answer only:"
    )


def build_prompt(
    tokenizer: Any,
    item: dict[str, Any],
    previous_response: str,
    max_trace_chars: int,
    retry_reason: str | None = None,
    first_attempt: str | None = None,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": build_user_prompt(item, previous_response, max_trace_chars, retry_reason, first_attempt),
        },
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    forced_think = "<|im_start|>assistant\n<think>\n"
    if text.endswith(forced_think):
        text = text[: -len(forced_think)] + "<|im_start|>assistant\n"
    return text


def load_transformers_model(model_id: str, adapter_dir: str | None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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


def load_vllm_model(args: argparse.Namespace) -> tuple[Any, Any, Any | None]:
    from transformers import AutoTokenizer

    tokenizer_source = args.adapter_dir or args.model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = make_llm(args.model_id, args, trust_remote_code=True, adapter_dir=args.adapter_dir)
    lora_request = make_lora_request(args.adapter_dir)
    return tokenizer, llm, lora_request


def finalize_record(
    row: dict[str, Any],
    item: dict[str, Any],
    response: str,
) -> dict[str, Any]:
    response = normalize_final_response(item, response)
    record = {
        "id": int(row["id"]),
        "is_mcq": is_mcq(item),
        "source_response": str(row.get("response", "")),
        "response": response,
        "answer_key": extract_answer_key(item, response, strict=True),
    }
    record.update(final_answer_diagnostics(item, response))
    for key in ("variant", "sample_index", "vote_count", "num_traces", "generated_tokens", "hit_token_limit"):
        if key in row:
            record[key] = row[key]
    return record


def bad_format_retry_reason(item: dict[str, Any], record: dict[str, Any], args: argparse.Namespace) -> str | None:
    if not args.retry_bad_format or is_mcq(item):
        return None
    expected = int(record.get("expected_slots") or 0)
    parsed = int(record.get("parsed_slots") or 0)
    if expected and parsed != expected:
        return f"it had {parsed} parsed answer slot(s), but the problem requires {expected}"
    return None


def mark_retry(record: dict[str, Any], first_response: str, reason: str) -> dict[str, Any]:
    record["first_finalizer_response"] = first_response
    record["retry_reason"] = reason
    return record


def finalize_with_transformers(
    to_finalize: list[dict[str, Any]],
    data_by_id: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    emit_record: RecordSink | None = None,
) -> list[dict[str, Any]]:
    import torch

    tokenizer, model = load_transformers_model(args.model_id, args.adapter_dir)
    finalized: list[dict[str, Any]] = []
    for row in tqdm(to_finalize, desc="Finalizing"):
        item = data_by_id[int(row["id"])]
        prompt = build_prompt(tokenizer, item, str(row.get("response", "")), args.max_trace_chars)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(model.device)
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
        record = finalize_record(row, item, response)
        reason = bad_format_retry_reason(item, record, args)
        if reason:
            retry_prompt = build_prompt(
                tokenizer,
                item,
                str(row.get("response", "")),
                args.max_trace_chars,
                retry_reason=reason,
                first_attempt=record["response"],
            )
            retry_inputs = tokenizer(
                retry_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(model.device)
            with torch.no_grad():
                retry_output = model.generate(
                    **retry_inputs,
                    max_new_tokens=args.retry_max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            retry_tokens = retry_output[0, retry_inputs["input_ids"].shape[1] :]
            retry_response = tokenizer.decode(retry_tokens, skip_special_tokens=True).strip()
            record = mark_retry(finalize_record(row, item, retry_response), record["response"], reason)
        finalized.append(record)
        if emit_record:
            emit_record(record)
    return finalized


def finalize_with_vllm(
    to_finalize: list[dict[str, Any]],
    data_by_id: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    emit_record: RecordSink | None = None,
) -> list[dict[str, Any]]:
    tokenizer, llm, lora_request = load_vllm_model(args)
    sampling_params = make_sampling_params(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )
    chunk_size = int(args.vllm_batch_size or len(to_finalize) or 1)
    finalized: list[dict[str, Any]] = []
    for batch in tqdm(list(batched(to_finalize, chunk_size)), desc="Finalizing"):
        prompts = []
        for row in batch:
            item = data_by_id[int(row["id"])]
            prompts.append(build_prompt(tokenizer, item, str(row.get("response", "")), args.max_trace_chars))
        outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request)
        batch_records: list[dict[str, Any]] = []
        retry_jobs: list[tuple[int, dict[str, Any], dict[str, Any], str]] = []
        for row, output in zip(batch, outputs):
            item = data_by_id[int(row["id"])]
            record = finalize_record(row, item, vllm_text(output))
            reason = bad_format_retry_reason(item, record, args)
            if reason:
                retry_jobs.append((len(batch_records), row, record, reason))
            batch_records.append(record)

        if retry_jobs:
            retry_prompts = [
                build_prompt(
                    tokenizer,
                    data_by_id[int(row["id"])],
                    str(row.get("response", "")),
                    args.max_trace_chars,
                    retry_reason=reason,
                    first_attempt=record["response"],
                )
                for _, row, record, reason in retry_jobs
            ]
            retry_sampling_params = make_sampling_params(
                max_tokens=args.retry_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
            )
            retry_outputs = llm.generate(retry_prompts, sampling_params=retry_sampling_params, lora_request=lora_request)
            for (record_index, row, record, reason), retry_output in zip(retry_jobs, retry_outputs):
                item = data_by_id[int(row["id"])]
                batch_records[record_index] = mark_retry(
                    finalize_record(row, item, vllm_text(retry_output)),
                    record["response"],
                    reason,
                )

        finalized.extend(batch_records)
        if emit_record:
            for record in batch_records:
                emit_record(record)
    return finalized


def score_record_in_place(
    record: dict[str, Any],
    data_by_id: dict[int, dict[str, Any]],
    score_judger: Judger | None,
) -> None:
    if score_judger is None:
        return
    item = data_by_id[int(record["id"])]
    if has_gold(item):
        record["gold"] = item["answer"]
        record["correct"] = score_item(score_judger, item, str(record.get("response", "")))


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    data_by_id = index_by_id(data)
    rows = read_jsonl(args.responses)
    output_path = Path(args.output)
    existing: list[dict[str, Any]] = read_jsonl_complete_prefix(output_path) if output_path.exists() else []
    seen_ids = {int(record["id"]) for record in existing}
    score_judger = Judger(strict_extract=False) if args.score else None

    def emit_record(record: dict[str, Any]) -> None:
        score_record_in_place(record, data_by_id, score_judger)
        append_jsonl_record(output_path, record)

    to_finalize: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for row in rows:
        item_id = int(row["id"])
        if item_id in seen_ids:
            continue
        item = data_by_id[int(row["id"])]
        strict_key = extract_answer_key(item, str(row.get("response", "")), strict=True)
        if args.only_missing and strict_key:
            record = {**row, "answer_key": strict_key}
            kept.append(record)
            emit_record(record)
        else:
            to_finalize.append(row)

    if existing:
        print(f"Resuming from {len(existing)} existing finalized rows in {output_path}.")
    print(f"Finalizing {len(to_finalize)} rows; keeping {len(kept)} rows unchanged.")
    finalized: list[dict[str, Any]] = existing + kept
    if to_finalize:
        if args.backend == "vllm":
            finalized.extend(finalize_with_vllm(to_finalize, data_by_id, args, emit_record))
        else:
            finalized.extend(finalize_with_transformers(to_finalize, data_by_id, args, emit_record))

    finalized.sort(key=lambda record: int(record["id"]))
    if args.score:
        for record in finalized:
            score_record_in_place(record, data_by_id, score_judger)
        print(json.dumps(summarize_results(finalized), indent=2, sort_keys=True))
    write_jsonl(args.output, finalized)
    print(f"Wrote finalized responses to {args.output}")


if __name__ == "__main__":
    main()
