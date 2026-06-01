# Added by Codex: evaluate a selector/finalizer LoRA on public traces with vLLM.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from judger import Judger
from math_comp.data import batched, has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.final_answer import final_answer_diagnostics, normalize_final_response
from math_comp.grpo_finalizer import build_finalizer_messages, direct_chat_prompt
from math_comp.prompts import format_options
from math_comp.scoring import extract_answer_key, score_item, summarize_results
from math_comp.vllm_helpers import add_vllm_cli_args, make_llm, make_lora_request, make_sampling_params, vllm_generated_tokens, vllm_text


MCQ_SYSTEM = (
    "You are an MCQ answer selector for a math competition. "
    "Use the problem, options, solver trace, and candidate answers to output exactly one uppercase option letter. "
    "Do not output option text. Do not explain."
)


def default_model_id() -> str:
    snapshot_root = REPO_ROOT / ".hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots"
    snapshots = [path for path in snapshot_root.glob("*") if path.is_dir()]
    if snapshots:
        return str(max(snapshots, key=lambda path: path.stat().st_mtime))
    return "Qwen/Qwen3-4B-Thinking-2507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a category LoRA adapter on raw traces.")
    parser.add_argument("--mode", choices=["mcq_selector", "finalizer_schema"], required=True)
    parser.add_argument("--data", default="data/public.jsonl")
    parser.add_argument("--raw-responses", required=True)
    parser.add_argument("--candidate-responses", nargs="*", default=[])
    parser.add_argument("--current-finalizer", default=None)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--model-id", default=default_model_id())
    parser.add_argument("--output", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-trace-chars", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--score", action="store_true")
    add_vllm_cli_args(parser, default_backend="vllm")
    return parser.parse_args()


def compact_trace(trace: str, max_chars: int) -> str:
    trace = str(trace or "").strip()
    if len(trace) <= max_chars:
        return trace
    head = max_chars // 4
    tail = max_chars - head
    omitted = len(trace) - head - tail
    return trace[:head].rstrip() + f"\n\n[... omitted {omitted} characters ...]\n\n" + trace[-tail:].lstrip()


def first_by_id(path: str | None) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        out.setdefault(int(row["id"]), row)
    return out


def rows_by_id(paths: list[str]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for path in paths:
        for row in read_jsonl(path):
            grouped.setdefault(int(row["id"]), []).append({**row, "source_path": path})
    return grouped


def candidate_summary(item: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = []
    for index, row in enumerate(rows[:12], start=1):
        response = str(row.get("response", ""))
        key = str(row.get("answer_key") or extract_answer_key(item, response, strict=True))
        lines.append(f"{index}. key={key or '<none>'}, tail={response[-220:]}")
    return "\n".join(lines) if lines else "<none>"


def mcq_messages(item: dict[str, Any], raw: dict[str, Any], candidates: list[dict[str, Any]], current: dict[str, Any] | None, max_trace_chars: int) -> list[dict[str, str]]:
    trace = compact_trace(str(raw.get("response", "")), max_trace_chars)
    raw_key = str(raw.get("answer_key") or extract_answer_key(item, trace, strict=True))
    current_response = str((current or {}).get("response", ""))
    current_key = str((current or {}).get("answer_key") or extract_answer_key(item, current_response, strict=True)) if current else ""
    user = (
        "Problem:\n"
        f"{item['question']}\n\n"
        "Options:\n"
        f"{format_options(item.get('options') or [])}\n\n"
        "Solver trace:\n"
        f"{trace}\n\n"
        "Raw extracted answer letter:\n"
        f"{raw_key or '<none>'}\n\n"
        "Current baseline/finalizer answer letter:\n"
        f"{current_key or '<none>'}\n\n"
        "Candidate answers:\n"
        f"{candidate_summary(item, candidates)}\n\n"
        "Final answer letter only:"
    )
    return [{"role": "system", "content": MCQ_SYSTEM}, {"role": "user", "content": user}]


def build_prompt(tokenizer: Any, args: argparse.Namespace, item: dict[str, Any], raw: dict[str, Any], candidates: list[dict[str, Any]], current: dict[str, Any] | None) -> str:
    if args.mode == "mcq_selector":
        messages = mcq_messages(item, raw, candidates, current, args.max_trace_chars)
    else:
        messages = build_finalizer_messages(
            item,
            str(raw.get("response", "")),
            raw_extracted_answer=str(raw.get("answer_key") or extract_answer_key(item, str(raw.get("response", "")), strict=True)),
            current_finalizer_answer=str((current or {}).get("response", "")),
            max_trace_chars=args.max_trace_chars,
        )
    return direct_chat_prompt(tokenizer, messages)


def normalize_output(item: dict[str, Any], mode: str, text: str) -> str:
    text = str(text).strip()
    if mode == "mcq_selector":
        key = extract_answer_key(item, text, strict=True)
        if key:
            return f"\\boxed{{{key.strip().upper()}}}"
        stripped = text.strip().upper()
        if len(stripped) == 1:
            return f"\\boxed{{{stripped}}}"
    return normalize_final_response(item, text)


def main() -> None:
    args = parse_args()
    if args.backend != "vllm":
        raise ValueError("64_eval_category_lora.py currently supports --backend vllm only.")

    data = read_jsonl(args.data)
    if args.offset:
        data = data[args.offset :]
    if args.limit is not None:
        data = data[: args.limit]
    raw_by_id = first_by_id(args.raw_responses)
    current_by_id = first_by_id(args.current_finalizer)
    candidate_by_id = rows_by_id(args.candidate_responses)
    selected_items = [
        item for item in data
        if int(item["id"]) in raw_by_id and ((args.mode == "mcq_selector" and is_mcq(item)) or (args.mode == "finalizer_schema" and not is_mcq(item)))
    ]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir if Path(args.adapter_dir).exists() else args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = make_llm(args.model_id, args, trust_remote_code=True, adapter_dir=args.adapter_dir)
    lora_request = make_lora_request(args.adapter_dir)
    sampling = make_sampling_params(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )
    prompts = [
        build_prompt(tokenizer, args, item, raw_by_id[int(item["id"])], candidate_by_id.get(int(item["id"]), []), current_by_id.get(int(item["id"])))
        for item in selected_items
    ]

    rows: list[dict[str, Any]] = []
    score_judger = Judger(strict_extract=False)
    chunk_size = int(args.vllm_batch_size or len(prompts) or 1)
    item_index = 0
    for prompt_chunk in tqdm(list(batched(prompts, chunk_size)), desc=f"Evaluating {args.mode}"):
        outputs = llm.generate(prompt_chunk, sampling_params=sampling, lora_request=lora_request)
        for output in outputs:
            item = selected_items[item_index]
            raw_text = vllm_text(output)
            response = normalize_output(item, args.mode, raw_text)
            row = {
                "id": int(item["id"]),
                "is_mcq": is_mcq(item),
                "mode": args.mode,
                "raw_model_output": raw_text,
                "response": response,
                "answer_key": extract_answer_key(item, response, strict=True),
                "generated_tokens": vllm_generated_tokens(output),
            }
            row.update(final_answer_diagnostics(item, response))
            if args.score and has_gold(item):
                row["gold"] = item["answer"]
                row["correct"] = score_item(score_judger, item, response)
            rows.append(row)
            item_index += 1

    write_jsonl(args.output, rows)
    if args.score:
        print(json.dumps(summarize_results(rows), indent=2, sort_keys=True))
    print(f"Wrote {len(rows)} {args.mode} predictions to {args.output}")


if __name__ == "__main__":
    main()
