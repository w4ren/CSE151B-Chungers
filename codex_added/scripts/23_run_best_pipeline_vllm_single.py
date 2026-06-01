# Added by Codex: single-process vLLM best pipeline; not part of the original starter repository.

from __future__ import annotations

import argparse
import importlib.util
import json
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
from math_comp.data import batched, has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.prompts import build_prompt_text
from math_comp.scoring import extract_answer_key, score_item, summarize_results
from math_comp.submission import write_submission_csv
from math_comp.variants import get_variant
from math_comp.vllm_helpers import (
    add_vllm_cli_args,
    make_llm,
    make_lora_request,
    make_sampling_params,
    vllm_generated_tokens,
    vllm_text,
)


def load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINALIZE = load_script_module("finalize_with_qwen", CODEX_ROOT / "scripts/16_finalize_with_qwen.py")
RERANK = load_script_module("rerank_with_qwen", CODEX_ROOT / "scripts/17_rerank_with_qwen.py")
SELECT = load_script_module("select_predictions", CODEX_ROOT / "scripts/18_select_predictions.py")


def default_model_path() -> str:
    snapshot_root = REPO_ROOT / ".hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots"
    snapshots = [path for path in snapshot_root.glob("*") if path.is_dir()]
    if snapshots:
        return str(max(snapshots, key=lambda path: path.stat().st_mtime))
    return "Qwen/Qwen3-4B-Thinking-2507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the best-known pipeline with one persistent vLLM engine.")
    parser.add_argument("--data", default="data/private.jsonl", help="Competition JSONL.")
    parser.add_argument("--out-dir", default="codex_added/results/best_private_vllm_single", help="Output directory.")
    parser.add_argument("--submission", default="codex_added/submissions/best_submission.csv", help="Submission CSV path.")
    parser.add_argument("--model-id", default=None, help="Model id or local snapshot path.")
    parser.add_argument("--adapter-dir", default="codex_added/models/qwen3_answer_format_lora", help="LoRA adapter directory.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N rows.")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows after offset.")
    parser.add_argument("--score", action="store_true", help="Score output when data has answers.")
    parser.add_argument("--write-submission", action="store_true", default=True, help="Write CSV for complete runs.")
    parser.add_argument("--no-write-submission", action="store_false", dest="write_submission", help="Skip CSV writing.")
    parser.add_argument("--max-trace-chars", type=int, default=5000, help="Tail chars kept for finalizer prompts.")
    parser.add_argument("--sweep-2048-tokens", type=int, default=2048, help="Generation budget for the long sweep stage.")
    parser.add_argument("--sweep-1024-tokens", type=int, default=1024, help="Generation budget for the short sweep stage.")
    parser.add_argument("--finalizer-tokens", type=int, default=64, help="Generation budget for final-answer formatting stages.")
    parser.add_argument(
        "--retry-bad-format",
        action="store_true",
        help="Retry free-form finalizer outputs whose parsed slot count is wrong.",
    )
    parser.add_argument(
        "--retry-max-new-tokens",
        type=int,
        default=128,
        help="Generation budget for stricter bad-format finalizer retries.",
    )
    parser.add_argument("--rerank-tokens", type=int, default=1536, help="Generation budget for conflict reranking.")
    parser.add_argument("--rerank-max-response-chars", type=int, default=800, help="Tail chars per final response.")
    parser.add_argument("--rerank-max-trace-chars", type=int, default=1200, help="Tail chars per source trace.")
    parser.add_argument("--rerank-assistant-mode", choices=["think", "direct"], default="think", help="Reranker prompt mode.")
    add_vllm_cli_args(parser, default_backend="vllm")
    return parser.parse_args()


def score_records(records: list[dict[str, Any]], data_by_id: dict[int, dict[str, Any]], enabled: bool) -> None:
    if not enabled:
        return
    judger = Judger(strict_extract=False)
    for record in records:
        item = data_by_id[int(record["id"])]
        if has_gold(item):
            record["gold"] = item["answer"]
            record["correct"] = score_item(judger, item, str(record.get("response", "")))
    print(json.dumps(summarize_results(records), indent=2, sort_keys=True))


def generate_chunks(llm: Any, prompts: list[str], sampling_params: Any, args: argparse.Namespace, lora_request: Any | None = None) -> list[Any]:
    if not prompts:
        return []
    chunk_size = int(args.vllm_batch_size or len(prompts))
    outputs: list[Any] = []
    for chunk in tqdm(list(batched(prompts, chunk_size)), desc="vLLM generate"):
        outputs.extend(llm.generate(chunk, sampling_params=sampling_params, lora_request=lora_request))
    return outputs


def run_sweep(
    llm: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    *,
    max_tokens: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    variant_name = "final_box_only"
    variant = get_variant(variant_name)
    prompts = [build_prompt_text(tokenizer, item, variant) for item in items]
    sampling_params = make_sampling_params(
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        do_sample=True,
    )
    outputs = generate_chunks(llm, prompts, sampling_params, args)
    rows: list[dict[str, Any]] = []
    for item, output in zip(items, outputs):
        response = vllm_text(output)
        generated_tokens = vllm_generated_tokens(output)
        rows.append(
            {
                "id": int(item["id"]),
                "is_mcq": is_mcq(item),
                "variant": variant_name,
                "sample_index": 0,
                "answer_key": extract_answer_key(item, response, strict=True),
                "generated_tokens": generated_tokens,
                "hit_token_limit": generated_tokens >= max_tokens,
                "response": response,
            }
        )
    return rows


def finalize_rows(
    llm: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    data_by_id: dict[int, dict[str, Any]],
    lora_request: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    prompts = [
        FINALIZE.build_prompt(tokenizer, data_by_id[int(row["id"])], str(row.get("response", "")), args.max_trace_chars)
        for row in rows
    ]
    sampling_params = make_sampling_params(
        max_tokens=args.finalizer_tokens,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    outputs = generate_chunks(llm, prompts, sampling_params, args, lora_request=lora_request)
    finalized: list[dict[str, Any]] = []
    retry_jobs: list[tuple[int, dict[str, Any], dict[str, Any], str]] = []
    for row, output in zip(rows, outputs):
        item = data_by_id[int(row["id"])]
        record = FINALIZE.finalize_record(row, item, vllm_text(output))
        reason = FINALIZE.bad_format_retry_reason(item, record, args)
        if reason:
            retry_jobs.append((len(finalized), row, record, reason))
        finalized.append(record)

    if retry_jobs:
        retry_prompts = [
            FINALIZE.build_prompt(
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
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
        )
        retry_outputs = generate_chunks(llm, retry_prompts, retry_sampling_params, args, lora_request=lora_request)
        for (record_index, row, record, reason), retry_output in zip(retry_jobs, retry_outputs):
            item = data_by_id[int(row["id"])]
            finalized[record_index] = FINALIZE.mark_retry(
                FINALIZE.finalize_record(row, item, vllm_text(retry_output)),
                record["response"],
                reason,
            )

    return finalized


def grouped_candidates(response_files: list[list[dict[str, Any]]], data_by_id: dict[int, dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for file_index, rows in enumerate(response_files):
        for row_index, row in enumerate(rows):
            item = data_by_id.get(int(row["id"]))
            if item is None:
                continue
            response = str(row.get("response", ""))
            key = str(row.get("answer_key") or extract_answer_key(item, response, strict=True))
            grouped[int(row["id"])].append({**row, "answer_key": key, "source_file": f"memory:{file_index}", "source_row": row_index})
    return grouped


def rerank_conflicts(
    llm: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    candidate_sets: list[list[dict[str, Any]]],
    data_by_id: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    grouped = grouped_candidates(candidate_sets, data_by_id)
    score_judger = Judger(strict_extract=False)
    rows: list[dict[str, Any] | None] = []
    jobs: list[dict[str, Any]] = []
    for item in tqdm(items, desc="Preparing rerank"):
        candidates = RERANK.select_candidates(item, grouped.get(int(item["id"]), []), 8, True)
        if candidates and len(RERANK.unique_answer_keys(candidates)) <= 1:
            rows.append(
                RERANK.make_row_from_response(
                    item,
                    str(candidates[0].get("response", "")),
                    candidates,
                    "kept_unanimous",
                    score_judger,
                    args.score,
                )
            )
            continue
        rows.append(None)
        jobs.append({"row_index": len(rows) - 1, "item": item, "candidates": candidates})

    prompts = [
        RERANK.build_prompt(
            tokenizer,
            job["item"],
            job["candidates"],
            args.rerank_max_response_chars,
            args.rerank_max_trace_chars,
            args.rerank_assistant_mode,
        )
        for job in jobs
    ]
    sampling_params = make_sampling_params(
        max_tokens=args.rerank_tokens,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    outputs = generate_chunks(llm, prompts, sampling_params, args)
    for job, output in zip(jobs, outputs):
        item = job["item"]
        candidates = job["candidates"]
        rows[int(job["row_index"])] = RERANK.make_row_from_response(
            item,
            vllm_text(output),
            candidates,
            "reranked",
            score_judger,
            args.score,
        )
    return [row for row in rows if row is not None]


def select_predictions(
    items: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    override_rows: list[dict[str, Any]],
    *,
    score: bool,
) -> list[dict[str, Any]]:
    base_by_id = index_by_id(base_rows)
    override_by_id = index_by_id(override_rows)
    score_judger = Judger(strict_extract=False)
    selected_rows: list[dict[str, Any]] = []
    for item in items:
        item_id = int(item["id"])
        base = base_by_id.get(item_id)
        override = override_by_id.get(item_id)
        if base is None and override is None:
            selected = {"id": item_id, "response": ""}
            source = "missing"
        elif override is not None and SELECT.should_use_override(item, "free_form_override"):
            selected = override
            source = "override"
        else:
            selected = base if base is not None else override
            source = "base" if base is not None else "override_fallback"
        selected_rows.append(SELECT.with_metadata(item, selected, source, score_judger, score))
    return selected_rows


def main() -> None:
    args = parse_args()
    args.backend = "vllm"
    args.model_id = args.model_id or default_model_path()

    data = read_jsonl(args.data)
    total_rows = len(data)
    if args.offset:
        data = data[args.offset :]
    if args.limit is not None:
        data = data[: args.limit]
    data_by_id = index_by_id(data)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using model: {args.model_id}")
    print(f"Running {len(data)} rows through one persistent vLLM engine.")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = make_llm(args.model_id, args, trust_remote_code=True, adapter_dir=args.adapter_dir)
    lora_request = make_lora_request(args.adapter_dir)

    sweep_2048 = run_sweep(llm, tokenizer, data, max_tokens=args.sweep_2048_tokens, args=args)
    write_jsonl(out_dir / "sweep_2048.jsonl", sweep_2048)

    finalized_2048 = finalize_rows(llm, tokenizer, sweep_2048, data_by_id, lora_request, args)
    score_records(finalized_2048, data_by_id, args.score)
    write_jsonl(out_dir / "finalized_2048.jsonl", finalized_2048)

    sweep_1024 = run_sweep(llm, tokenizer, data, max_tokens=args.sweep_1024_tokens, args=args)
    write_jsonl(out_dir / "sweep_1024.jsonl", sweep_1024)

    finalized_1024 = finalize_rows(llm, tokenizer, sweep_1024, data_by_id, lora_request, args)
    score_records(finalized_1024, data_by_id, args.score)
    write_jsonl(out_dir / "finalized_1024.jsonl", finalized_1024)

    reranked = rerank_conflicts(llm, tokenizer, data, [finalized_1024, finalized_2048], data_by_id, args)
    score_records(reranked, data_by_id, args.score)
    write_jsonl(out_dir / "reranked_conflicts.jsonl", reranked)

    finalized_reranked = finalize_rows(llm, tokenizer, reranked, data_by_id, lora_request, args)
    score_records(finalized_reranked, data_by_id, args.score)
    write_jsonl(out_dir / "finalized_reranked_conflicts.jsonl", finalized_reranked)

    selected = select_predictions(data, finalized_2048, finalized_reranked, score=args.score)
    score_records(selected, data_by_id, args.score)
    write_jsonl(out_dir / "selected.jsonl", selected)
    print(f"Wrote selected JSONL to {out_dir / 'selected.jsonl'}")

    complete_run = args.offset == 0 and args.limit is None and len(data) == total_rows
    if args.write_submission and complete_run:
        write_submission_csv(args.data, out_dir / "selected.jsonl", args.submission)
        print(f"Wrote submission CSV to {args.submission}")
    else:
        print("Submission CSV skipped; this was a partial run or --no-write-submission was used.")


if __name__ == "__main__":
    main()
