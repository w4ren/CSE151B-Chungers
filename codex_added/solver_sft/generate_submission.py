# Added by Codex: generate test-set predictions with solver LoRA plus optional finalizer.

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_comp.data import index_by_id, write_jsonl

from eval_solver import generate_solver_outputs, read_items, run_finalizer


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


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    model_defaults = section(defaults, "model")
    generation_defaults = section(defaults, "generation")
    finalizer_defaults = section(defaults, "finalizer")
    path_defaults = section(defaults, "paths")
    submission_defaults = section(defaults, "submission")

    parser = argparse.ArgumentParser(description="Generate submission CSV with a solver LoRA adapter.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--test_file", "--test-file", default=path_defaults.get("test_file", "data/private.jsonl"))
    parser.add_argument("--model_name", "--model-name", default=model_defaults.get("name", "Qwen/Qwen3-4B-Thinking-2507"))
    parser.add_argument("--solver_adapter", "--solver-adapter", required=not bool(path_defaults.get("solver_adapter")), default=path_defaults.get("solver_adapter"))
    parser.add_argument("--answer_finalizer_adapter", "--answer-finalizer-adapter", default=path_defaults.get("answer_finalizer_adapter", "codex_added/solver_sft/adapters/answer_finalizer_lora"))
    parser.add_argument("--without_finalizer", "--without-finalizer", action="store_true", default=not bool(finalizer_defaults.get("enabled", True)))
    parser.add_argument("--output_csv", "--output-csv", default=path_defaults.get("submission_csv", "codex_added/solver_sft/submissions/solver_sft_submission.csv"))
    parser.add_argument("--output_jsonl", "--output-jsonl", default=path_defaults.get("submission_jsonl", "codex_added/solver_sft/results/solver_sft_submission.jsonl"))
    parser.add_argument("--answer_column", "--answer-column", default=submission_defaults.get("answer_column", "answer"))
    parser.add_argument("--id_column", "--id-column", default=submission_defaults.get("id_column", "id"))
    parser.add_argument("--limit", type=int, default=submission_defaults.get("limit", None))
    parser.add_argument("--offset", type=int, default=int(submission_defaults.get("offset", 0)))
    parser.add_argument("--variant", default=generation_defaults.get("variant", "final_box_only"))
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=int(generation_defaults.get("max_new_tokens", 8192)))
    parser.add_argument("--max_input_tokens", "--max-input-tokens", type=int, default=int(generation_defaults.get("max_input_tokens", 4096)))
    parser.add_argument("--temperature", type=float, default=float(generation_defaults.get("temperature", 0.6)))
    parser.add_argument("--top_p", "--top-p", type=float, default=float(generation_defaults.get("top_p", 0.95)))
    parser.add_argument("--top_k", "--top-k", type=int, default=int(generation_defaults.get("top_k", 20)))
    parser.add_argument("--batch_size", "--batch-size", type=int, default=int(generation_defaults.get("batch_size", 1)))
    parser.add_argument("--device_map", "--device-map", default=model_defaults.get("device_map", "auto"))
    parser.add_argument("--use_4bit", "--use-4bit", action=argparse.BooleanOptionalAction, default=bool(submission_defaults.get("use_4bit", False)))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=bool(submission_defaults.get("bf16", True)))
    parser.add_argument("--finalizer_max_input_tokens", "--finalizer-max-input-tokens", type=int, default=int(finalizer_defaults.get("max_input_tokens", 4096)))
    parser.add_argument("--finalizer_max_new_tokens", "--finalizer-max-new-tokens", type=int, default=int(finalizer_defaults.get("max_new_tokens", 64)))
    parser.add_argument("--finalizer_max_trace_chars", "--finalizer-max-trace-chars", type=int, default=int(finalizer_defaults.get("max_trace_chars", 6000)))
    parser.add_argument("--finalizer_temperature", "--finalizer-temperature", type=float, default=float(finalizer_defaults.get("temperature", 0.1)))
    parser.add_argument("--finalizer_top_p", "--finalizer-top-p", type=float, default=float(finalizer_defaults.get("top_p", 0.9)))
    parser.add_argument("--retry_bad_format", "--retry-bad-format", action=argparse.BooleanOptionalAction, default=bool(finalizer_defaults.get("retry_bad_format", True)))
    parser.add_argument("--retry_max_new_tokens", "--retry-max-new-tokens", type=int, default=int(finalizer_defaults.get("retry_max_new_tokens", 128)))
    return parser


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, remaining = pre.parse_known_args()
    config = load_yaml(known.config)
    parser = build_parser(config)
    return parser.parse_args(["--config", known.config] + remaining if known.config else remaining)


def write_submission_csv(path: str | Path, items: list[dict[str, Any]], records: list[dict[str, Any]], id_column: str, answer_column: str) -> None:
    pred_by_id = index_by_id(records)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_column, answer_column])
        writer.writeheader()
        for item in items:
            item_id = int(item["id"])
            response = str(pred_by_id[item_id].get("response", ""))
            writer.writerow({id_column: item_id, answer_column: response})


def main() -> None:
    args = parse_args()
    items = read_items(args.test_file)
    if args.offset:
        items = items[args.offset :]
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise RuntimeError("No test rows to run.")

    raw_records = generate_solver_outputs(items, args)
    data_by_id = index_by_id(items)
    final_records = raw_records if args.without_finalizer else run_finalizer(raw_records, data_by_id, args)
    final_records.sort(key=lambda row: int(row["id"]))

    write_jsonl(args.output_jsonl, final_records)
    write_submission_csv(args.output_csv, items, final_records, args.id_column, args.answer_column)
    print(json.dumps({"rows": len(final_records), "csv": args.output_csv, "jsonl": args.output_jsonl}, indent=2))


if __name__ == "__main__":
    main()
