#!/usr/bin/env python3
# Added by Codex: targeted MCQ trace extraction and option logprob scoring experiment.

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from judger import Judger
from math_comp.data import has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.prompts import format_options
from math_comp.scoring import extract_answer_key, score_item
from math_comp.vllm_helpers import add_vllm_cli_args, make_llm, make_sampling_params, vllm_generated_tokens, vllm_text

SYSTEM = (
    "You are an expert multiple-choice answer extractor. Use the problem, choices, and the "
    "previous reasoning trace to choose the single option letter best supported by the work. "
    "Do not solve from scratch unless the trace is unusable. If the trace changes its mind, "
    "prefer the final consistent calculation. If the trace loops, ignore repeated checks. "
    "Return exactly one boxed option letter."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCQ trace extractor and option logprob scorer for capped V6 failures.")
    parser.add_argument("--data", required=True, help="Full public200 data JSONL.")
    parser.add_argument("--raw-responses", required=True, help="Raw 8k trace JSONL.")
    parser.add_argument("--baseline", required=True, help="V6 baseline JSONL.")
    parser.add_argument("--current-finalized", default=None, help="Existing LoRA finalizer JSONL for baseline comparison.")
    parser.add_argument("--output", default=None, help="Output JSONL for method predictions.")
    parser.add_argument("--summary-output", default=None, help="Output JSON summary.")
    parser.add_argument("--details-output", default=None, help="Output JSONL with per-target baseline/method details.")
    parser.add_argument("--predictions", nargs="*", default=None, help="Prediction shards to summarize instead of generating.")
    parser.add_argument("--prepare-target-data", default=None, help="Write the target data slice and exit if --prepare-only.")
    parser.add_argument("--prepare-target-raw", default=None, help="Write the target raw trace slice and exit if --prepare-only.")
    parser.add_argument("--prepare-only", action="store_true", help="Only build target slices and summary inputs.")
    parser.add_argument("--target-ids", default=None, help="Comma-separated ids. Default: V6-wrong MCQs whose raw trace hit cap.")
    parser.add_argument("--offset", type=int, default=0, help="Offset into selected target ids for sharding.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected target ids for sharding.")
    parser.add_argument("--methods", default="extractor_letter,logprob", help="Comma-separated: extractor,extractor_letter,logprob.")
    parser.add_argument("--max-trace-chars", type=int, default=18000, help="Head+tail trace character budget.")
    parser.add_argument("--trace-head-frac", type=float, default=0.35, help="Fraction of trace budget reserved for the trace head.")
    parser.add_argument("--extractor-max-new-tokens", type=int, default=256, help="Extractor generation budget.")
    parser.add_argument("--prompt-token-margin", type=int, default=32, help="Safety margin below max_model_len after tokenized prompt fitting.")
    parser.add_argument("--extractor-temperature", type=float, default=0.1, help="Extractor temperature.")
    parser.add_argument("--extractor-top-p", type=float, default=0.9, help="Extractor top-p.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507", help="Model id or local snapshot path.")
    add_vllm_cli_args(parser, default_backend="vllm")
    return parser.parse_args()


def compact_trace(trace: str, max_chars: int, head_frac: float) -> str:
    trace = trace.strip()
    if max_chars <= 0 or len(trace) <= max_chars:
        return trace
    head_chars = max(0, min(max_chars, int(max_chars * head_frac)))
    tail_chars = max_chars - head_chars
    head = trace[:head_chars].rstrip()
    tail = trace[-tail_chars:].lstrip() if tail_chars else ""
    omitted = len(trace) - len(head) - len(tail)
    return f"{head}\n\n[... omitted {omitted} characters from the middle of the capped trace ...]\n\n{tail}"


def target_ids_from_inputs(
    data: list[dict[str, Any]],
    raw_by_id: dict[int, dict[str, Any]],
    baseline_by_id: dict[int, dict[str, Any]],
    requested: str | None,
) -> list[int]:
    if requested:
        wanted = [int(part.strip()) for part in requested.split(",") if part.strip()]
        valid = {int(item["id"]) for item in data}
        missing = [item_id for item_id in wanted if item_id not in valid]
        if missing:
            raise ValueError(f"Requested target ids are not in data: {missing}")
        return wanted

    ids: list[int] = []
    for item in data:
        item_id = int(item["id"])
        base = baseline_by_id.get(item_id)
        raw = raw_by_id.get(item_id)
        if not base or not raw:
            continue
        if is_mcq(item) and bool(raw.get("hit_token_limit")) and not bool(base.get("correct")):
            ids.append(item_id)
    return ids


def question_block(item: dict[str, Any]) -> str:
    text = str(item["question"])
    options = item.get("options") or []
    if options:
        text += "\n\nOptions:\n" + format_options(options)
    return text


def extractor_user_prompt(item: dict[str, Any], trace: str) -> str:
    return (
        "Problem and choices:\n"
        f"{question_block(item)}\n\n"
        "Previous reasoning trace from the solver. The trace may be capped or unfinished:\n"
        f"{trace}\n\n"
        "Task:\n"
        "- Extract the single option letter best supported by the trace.\n"
        "- Do not reward a later loop that contradicts a completed calculation without new evidence.\n"
        "- If several option letters are mentioned, choose the one supported by the final consistent reasoning.\n"
        "- Return only one final answer in the form \\boxed{A}.\n\n"
        "Final answer only:"
    )


def logprob_user_prompt(item: dict[str, Any], trace: str) -> str:
    return (
        "Problem and choices:\n"
        f"{question_block(item)}\n\n"
        "Previous reasoning trace from the solver. The trace may be capped or unfinished:\n"
        f"{trace}\n\n"
        "Choose the single option letter best supported by the trace. "
        "Answer with exactly one boxed option letter."
    )


def direct_chat_prompt(tokenizer: Any, system: str, user: str, assistant_prefix: str = "") -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    forced = "<|im_start|>assistant\n<think>\n"
    direct = "<|im_start|>assistant\n"
    if text.endswith(forced):
        text = text[: -len(forced)] + direct
    return text + assistant_prefix


def fit_direct_prompt(
    tokenizer: Any,
    system: str,
    item: dict[str, Any],
    raw_trace: str,
    user_builder: Any,
    assistant_prefix: str,
    max_trace_chars: int,
    head_frac: float,
    max_model_len: int | None,
    max_new_tokens: int,
    margin: int,
) -> tuple[str, int, int]:
    if max_model_len is None:
        max_model_len = 8192
    prompt_budget = max(256, int(max_model_len) - int(max_new_tokens) - int(margin))
    trace_budget = max_trace_chars
    last_prompt = ""
    last_trace_chars = 0
    last_tokens = 0
    while trace_budget >= 512:
        trace = compact_trace(raw_trace, trace_budget, head_frac)
        prompt = direct_chat_prompt(tokenizer, system, user_builder(item, trace), assistant_prefix=assistant_prefix)
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        last_prompt, last_trace_chars, last_tokens = prompt, len(trace), tokens
        if tokens <= prompt_budget:
            return prompt, len(trace), tokens
        shrink = max(512, int(trace_budget * 0.82))
        if shrink >= trace_budget:
            shrink = trace_budget - 512
        trace_budget = shrink
    return last_prompt, last_trace_chars, last_tokens


def letter_token_ids(tokenizer: Any, option_count: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for index in range(option_count):
        letter = chr(65 + index)
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Option letter {letter!r} is not one token: {ids}")
        out[letter] = int(ids[0])
    return out


def logprob_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("logprob", "log_prob"):
            if key in value:
                return float(value[key])
    for attr in ("logprob", "log_prob"):
        if hasattr(value, attr):
            return float(getattr(value, attr))
    return None


def score_record(item: dict[str, Any], response: str, judger: Judger) -> tuple[str, bool | None]:
    key = extract_answer_key(item, response, strict=True)
    correct = score_item(judger, item, response) if has_gold(item) else None
    return key, correct


def build_base_detail(
    item: dict[str, Any],
    raw: dict[str, Any],
    baseline: dict[str, Any],
    current_finalized: dict[str, Any] | None,
    judger: Judger,
) -> dict[str, Any]:
    raw_response = str(raw.get("response", ""))
    raw_key, raw_correct = score_record(item, raw_response, judger)
    detail: dict[str, Any] = {
        "id": int(item["id"]),
        "gold": item.get("answer"),
        "raw_answer_key": raw_key,
        "raw_correct": raw_correct,
        "raw_generated_tokens": int(raw.get("generated_tokens") or 0),
        "raw_hit_token_limit": bool(raw.get("hit_token_limit")),
        "v6_answer_key": baseline.get("answer_key") or extract_answer_key(item, str(baseline.get("response", "")), strict=True),
        "v6_correct": bool(baseline.get("correct")),
        "v6_response": baseline.get("response", ""),
    }
    if current_finalized is not None:
        cur_response = str(current_finalized.get("response", ""))
        cur_key, cur_correct = score_record(item, cur_response, judger)
        detail.update(
            {
                "current_finalizer_answer_key": cur_key,
                "current_finalizer_correct": cur_correct,
                "current_finalizer_response": cur_response,
            }
        )
    return detail


def run_generation(args: argparse.Namespace) -> None:
    if args.backend != "vllm":
        raise ValueError("This experiment currently supports --backend vllm only.")
    if not args.output:
        raise ValueError("--output is required when generating predictions.")

    from transformers import AutoTokenizer

    data = read_jsonl(args.data)
    raw_by_id = index_by_id(read_jsonl(args.raw_responses))
    baseline_by_id = index_by_id(read_jsonl(args.baseline))
    target_ids = target_ids_from_inputs(data, raw_by_id, baseline_by_id, args.target_ids)
    selected_ids = target_ids[args.offset :]
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]
    data_by_id = index_by_id(data)
    methods = {method.strip() for method in args.methods.split(",") if method.strip()}
    unknown = methods - {"extractor", "extractor_letter", "logprob"}
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = make_llm(args.model_id, args, trust_remote_code=True, adapter_dir=None)
    judger = Judger(strict_extract=False)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    if "extractor" in methods:
        prompts = []
        jobs = []
        for item_id in selected_ids:
            item = data_by_id[item_id]
            raw = raw_by_id[item_id]
            prompt, trace_chars, prompt_tokens = fit_direct_prompt(
                tokenizer,
                SYSTEM,
                item,
                str(raw.get("response", "")),
                extractor_user_prompt,
                "",
                args.max_trace_chars,
                args.trace_head_frac,
                args.max_model_len,
                args.extractor_max_new_tokens,
                args.prompt_token_margin,
            )
            prompts.append(prompt)
            jobs.append((item, raw, trace_chars, prompt_tokens))
        sampling = make_sampling_params(
            max_tokens=args.extractor_max_new_tokens,
            temperature=args.extractor_temperature,
            top_p=args.extractor_top_p,
            do_sample=args.extractor_temperature > 0,
        )
        for (item, raw, trace_chars, prompt_tokens), output in zip(
            jobs,
            llm.generate(prompts, sampling_params=sampling, use_tqdm=True),
        ):
            response = vllm_text(output)
            key, correct = score_record(item, response, judger)
            rows.append(
                {
                    "id": int(item["id"]),
                    "is_mcq": True,
                    "method": "extractor",
                    "response": response,
                    "answer_key": key,
                    "gold": item.get("answer"),
                    "correct": correct,
                    "generated_tokens": vllm_generated_tokens(output),
                    "hit_token_limit": vllm_generated_tokens(output) >= args.extractor_max_new_tokens,
                    "raw_generated_tokens": int(raw.get("generated_tokens") or 0),
                    "raw_hit_token_limit": bool(raw.get("hit_token_limit")),
                    "trace_chars_used": trace_chars,
                    "prompt_tokens": prompt_tokens,
                }
            )

    if "extractor_letter" in methods:
        prompts = []
        jobs = []
        params = []
        for item_id in selected_ids:
            item = data_by_id[item_id]
            raw = raw_by_id[item_id]
            token_by_letter = letter_token_ids(tokenizer, len(item.get("options") or []))
            prompt, trace_chars, prompt_tokens = fit_direct_prompt(
                tokenizer,
                SYSTEM,
                item,
                str(raw.get("response", "")),
                extractor_user_prompt,
                "\\boxed{",
                args.max_trace_chars,
                args.trace_head_frac,
                args.max_model_len,
                1,
                args.prompt_token_margin,
            )
            prompts.append(prompt)
            params.append(
                make_sampling_params(
                    max_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    allowed_token_ids=list(token_by_letter.values()),
                )
            )
            jobs.append((item, raw, token_by_letter, trace_chars, prompt_tokens))
        for (item, raw, token_by_letter, trace_chars, prompt_tokens), output in zip(
            jobs,
            llm.generate(prompts, sampling_params=params, use_tqdm=True),
        ):
            text = vllm_text(output)
            chosen = text.strip()[:1].upper() if text.strip() else ""
            response = f"\\boxed{{{chosen}}}" if chosen in token_by_letter else text
            key, correct = score_record(item, response, judger)
            rows.append(
                {
                    "id": int(item["id"]),
                    "is_mcq": True,
                    "method": "extractor_letter",
                    "response": response,
                    "generated_text": text,
                    "answer_key": key,
                    "gold": item.get("answer"),
                    "correct": correct,
                    "generated_tokens": vllm_generated_tokens(output),
                    "hit_token_limit": False,
                    "raw_generated_tokens": int(raw.get("generated_tokens") or 0),
                    "raw_hit_token_limit": bool(raw.get("hit_token_limit")),
                    "trace_chars_used": trace_chars,
                    "prompt_tokens": prompt_tokens,
                }
            )

    if "logprob" in methods:
        prompts = []
        jobs = []
        params = []
        for item_id in selected_ids:
            item = data_by_id[item_id]
            raw = raw_by_id[item_id]
            token_by_letter = letter_token_ids(tokenizer, len(item.get("options") or []))
            prompt, trace_chars, prompt_tokens = fit_direct_prompt(
                tokenizer,
                SYSTEM,
                item,
                str(raw.get("response", "")),
                logprob_user_prompt,
                "\\boxed{",
                args.max_trace_chars,
                args.trace_head_frac,
                args.max_model_len,
                1,
                args.prompt_token_margin,
            )
            prompts.append(prompt)
            params.append(
                make_sampling_params(
                    max_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    allowed_token_ids=list(token_by_letter.values()),
                )
            )
            params[-1].logprobs = max(5, len(token_by_letter))
            params[-1].logprob_token_ids = list(token_by_letter.values())
            jobs.append((item, raw, token_by_letter, trace_chars, prompt_tokens))
        for (item, raw, token_by_letter, trace_chars, prompt_tokens), output in zip(
            jobs,
            llm.generate(prompts, sampling_params=params, use_tqdm=True),
        ):
            text = vllm_text(output)
            generated_letter = text.strip()[:1].upper() if text.strip() else ""
            logprobs = getattr(output.outputs[0], "logprobs", None) or []
            first_logprobs = logprobs[0] if logprobs else {}
            option_logprobs: dict[str, float | None] = {}
            for letter, token_id in token_by_letter.items():
                option_logprobs[letter] = logprob_value(first_logprobs.get(token_id) if hasattr(first_logprobs, "get") else None)
            scored_options = {letter: value for letter, value in option_logprobs.items() if value is not None}
            if scored_options:
                chosen = max(scored_options.items(), key=lambda pair: pair[1])[0]
            elif generated_letter in token_by_letter:
                chosen = generated_letter
            else:
                chosen = ""
            response = f"\\boxed{{{chosen}}}" if chosen else text
            key, correct = score_record(item, response, judger)
            rows.append(
                {
                    "id": int(item["id"]),
                    "is_mcq": True,
                    "method": "option_logprob",
                    "response": response,
                    "generated_text": text,
                    "answer_key": key,
                    "gold": item.get("answer"),
                    "correct": correct,
                    "generated_tokens": vllm_generated_tokens(output),
                    "hit_token_limit": False,
                    "raw_generated_tokens": int(raw.get("generated_tokens") or 0),
                    "raw_hit_token_limit": bool(raw.get("hit_token_limit")),
                    "trace_chars_used": trace_chars,
                    "prompt_tokens": prompt_tokens,
                    "option_logprobs": option_logprobs,
                }
            )

    rows.sort(key=lambda row: (int(row["id"]), str(row["method"])))
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} method rows for {len(selected_ids)} ids to {out_path}")


def summarize(args: argparse.Namespace) -> None:
    data = read_jsonl(args.data)
    raw_by_id = index_by_id(read_jsonl(args.raw_responses))
    baseline_by_id = index_by_id(read_jsonl(args.baseline))
    current_by_id = index_by_id(read_jsonl(args.current_finalized)) if args.current_finalized else {}
    target_ids = target_ids_from_inputs(data, raw_by_id, baseline_by_id, args.target_ids)
    data_by_id = index_by_id(data)
    judger = Judger(strict_extract=False)

    prediction_rows: list[dict[str, Any]] = []
    if args.predictions:
        for path in args.predictions:
            prediction_rows.extend(read_jsonl(path))
    elif args.output and Path(args.output).exists():
        prediction_rows.extend(read_jsonl(args.output))

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if int(row["id"]) in set(target_ids):
            by_method[str(row.get("method", "unknown"))].append(row)

    details_by_id: dict[int, dict[str, Any]] = {}
    for item_id in target_ids:
        item = data_by_id[item_id]
        details_by_id[item_id] = build_base_detail(
            item,
            raw_by_id[item_id],
            baseline_by_id[item_id],
            current_by_id.get(item_id),
            judger,
        )

    def baseline_summary(label: str, key: str) -> dict[str, Any]:
        rows = [details_by_id[item_id] for item_id in target_ids]
        correct_ids = [int(row["id"]) for row in rows if row.get(key)]
        return {
            "label": label,
            "correct": len(correct_ids),
            "total": len(rows),
            "accuracy": len(correct_ids) / len(rows) if rows else 0.0,
            "correct_ids": correct_ids,
            "wrong_ids": [int(row["id"]) for row in rows if not row.get(key)],
        }

    method_summaries: dict[str, dict[str, Any]] = {}
    for method, rows in sorted(by_method.items()):
        seen: dict[int, dict[str, Any]] = {}
        for row in rows:
            seen[int(row["id"])] = row
        ordered = [seen[item_id] for item_id in target_ids if item_id in seen]
        correct_ids = [int(row["id"]) for row in ordered if bool(row.get("correct"))]
        missing_ids = [item_id for item_id in target_ids if item_id not in seen]
        gains_vs_v6 = [
            int(row["id"])
            for row in ordered
            if bool(row.get("correct")) and not bool(details_by_id[int(row["id"])].get("v6_correct"))
        ]
        losses_vs_v6 = [
            int(row["id"])
            for row in ordered
            if not bool(row.get("correct")) and bool(details_by_id[int(row["id"])].get("v6_correct"))
        ]
        net_vs_v6 = len(gains_vs_v6) - len(losses_vs_v6)
        method_summaries[method] = {
            "correct": len(correct_ids),
            "total": len(ordered),
            "target_total": len(target_ids),
            "accuracy": len(correct_ids) / len(ordered) if ordered else 0.0,
            "correct_ids": correct_ids,
            "wrong_ids": [int(row["id"]) for row in ordered if not bool(row.get("correct"))],
            "missing_ids": missing_ids,
            "gains_vs_v6": gains_vs_v6,
            "losses_vs_v6": losses_vs_v6,
            "net_vs_v6": net_vs_v6,
            "answer_key_rate": sum(bool(row.get("answer_key")) for row in ordered) / len(ordered) if ordered else 0.0,
            "hit_token_limit_rate": sum(bool(row.get("hit_token_limit")) for row in ordered) / len(ordered) if ordered else 0.0,
            "projected_public200_if_overridden_on_targets": {
                "correct": 158 + net_vs_v6,
                "total": 200,
            },
            "projected_mcq_if_overridden_on_targets": {
                "correct": 57 + net_vs_v6,
                "total": 78,
            },
        }
        for row in ordered:
            details_by_id[int(row["id"])][method] = {
                "answer_key": row.get("answer_key"),
                "correct": bool(row.get("correct")),
                "response": row.get("response", ""),
                "generated_tokens": row.get("generated_tokens"),
                "hit_token_limit": row.get("hit_token_limit"),
                "option_logprobs": row.get("option_logprobs"),
            }

    summary = {
        "target_ids": target_ids,
        "target_count": len(target_ids),
        "target_definition": "V6-wrong MCQs whose raw 8k trace hit the token cap" if not args.target_ids else "user-specified ids",
        "baselines": {
            "raw_8k_strict": baseline_summary("raw_8k_strict", "raw_correct"),
            "v6_current": baseline_summary("v6_current", "v6_correct"),
        },
        "methods": method_summaries,
    }
    if args.current_finalized:
        summary["baselines"]["current_lora_finalizer_existing"] = baseline_summary(
            "current_lora_finalizer_existing", "current_finalizer_correct"
        )

    if args.prepare_target_data:
        write_jsonl(args.prepare_target_data, [data_by_id[item_id] for item_id in target_ids])
    if args.prepare_target_raw:
        write_jsonl(args.prepare_target_raw, [raw_by_id[item_id] for item_id in target_ids])
    if args.details_output:
        write_jsonl(args.details_output, [details_by_id[item_id] for item_id in target_ids])
    if args.summary_output:
        out = Path(args.summary_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.prepare_only:
        summarize(args)
        return
    if args.predictions:
        summarize(args)
        return
    run_generation(args)
    if args.summary_output or args.details_output:
        summarize(args)


if __name__ == "__main__":
    main()
