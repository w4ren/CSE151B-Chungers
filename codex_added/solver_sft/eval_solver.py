# Added by Codex: evaluate a solver LoRA with the existing answer finalizer.

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from judger import Judger
from math_comp.data import answer_slot_count, batched, has_gold, index_by_id, is_mcq, write_jsonl
from math_comp.final_answer import final_answer_diagnostics
from math_comp.prompts import build_prompt_text
from math_comp.scoring import extract_answer_key, score_item, summarize_results
from math_comp.variants import get_variant


def load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINALIZE = load_script_module("finalize_with_qwen", CODEX_ROOT / "scripts/16_finalize_with_qwen.py")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def read_csv_items(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if "id" in item:
            item["id"] = int(item["id"])
        if "options" in item:
            parsed = parse_maybe_json(item["options"])
            if isinstance(parsed, str):
                if "||" in parsed:
                    parsed = [part.strip() for part in parsed.split("||") if part.strip()]
                elif "\n" in parsed:
                    parsed = [part.strip() for part in parsed.splitlines() if part.strip()]
            item["options"] = parsed if isinstance(parsed, list) else []
        if "answer" in item:
            item["answer"] = parse_maybe_json(item["answer"])
        if "question" not in item and "problem" in item:
            item["question"] = item["problem"]
        items.append(item)
    return items


def read_items(path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path_obj)
    if suffix == ".csv":
        return read_csv_items(path_obj)
    if suffix == ".json":
        with path_obj.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "rows", "examples"):
                if isinstance(data.get(key), list):
                    return data[key]
    raise ValueError(f"Unsupported diagnostic format: {path}")


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
    eval_defaults = section(defaults, "evaluation")
    generation_defaults = section(defaults, "generation")
    finalizer_defaults = section(defaults, "finalizer")
    path_defaults = section(defaults, "paths")

    parser = argparse.ArgumentParser(description="Evaluate Qwen solver LoRA on a labeled diagnostic set.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_file", "--data-file", default=eval_defaults.get("data_file", "codex_added/job_data/public_stratified_40.jsonl"))
    parser.add_argument("--model_name", "--model-name", default=model_defaults.get("name", "Qwen/Qwen3-4B-Thinking-2507"))
    parser.add_argument("--solver_adapter", "--solver-adapter", required=not bool(path_defaults.get("solver_adapter")), default=path_defaults.get("solver_adapter"))
    parser.add_argument("--answer_finalizer_adapter", "--answer-finalizer-adapter", default=path_defaults.get("answer_finalizer_adapter", "codex_added/solver_sft/adapters/answer_finalizer_lora"))
    parser.add_argument("--without_finalizer", "--without-finalizer", action="store_true", default=not bool(finalizer_defaults.get("enabled", True)))
    parser.add_argument("--output_raw", "--output-raw", default=path_defaults.get("eval_raw_output", "codex_added/solver_sft/results/eval_solver_raw.jsonl"))
    parser.add_argument("--output_final", "--output-final", default=path_defaults.get("eval_final_output", "codex_added/solver_sft/results/eval_solver_finalized.jsonl"))
    parser.add_argument("--limit", type=int, default=eval_defaults.get("limit", None))
    parser.add_argument("--offset", type=int, default=int(eval_defaults.get("offset", 0)))
    parser.add_argument("--variant", default=generation_defaults.get("variant", "final_box_only"))
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=int(generation_defaults.get("max_new_tokens", 8192)))
    parser.add_argument("--max_input_tokens", "--max-input-tokens", type=int, default=int(generation_defaults.get("max_input_tokens", 4096)))
    parser.add_argument("--temperature", type=float, default=float(generation_defaults.get("temperature", 0.6)))
    parser.add_argument("--top_p", "--top-p", type=float, default=float(generation_defaults.get("top_p", 0.95)))
    parser.add_argument("--top_k", "--top-k", type=int, default=int(generation_defaults.get("top_k", 20)))
    parser.add_argument("--batch_size", "--batch-size", type=int, default=int(generation_defaults.get("batch_size", 1)))
    parser.add_argument("--device_map", "--device-map", default=model_defaults.get("device_map", "auto"))
    parser.add_argument("--use_4bit", "--use-4bit", action=argparse.BooleanOptionalAction, default=bool(eval_defaults.get("use_4bit", False)))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=bool(eval_defaults.get("bf16", True)))
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


def load_solver(args: argparse.Namespace) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": args.device_map,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
    }
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model = PeftModel.from_pretrained(model, args.solver_adapter)
    model.eval()
    return tokenizer, model


def generate_solver_outputs(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    tokenizer, model = load_solver(args)
    variant = get_variant(args.variant)
    records: list[dict[str, Any]] = []
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "pad_token_id": tokenizer.eos_token_id,
    }
    for batch in tqdm(list(batched(items, args.batch_size)), desc="Solver generate"):
        prompts = [build_prompt_text(tokenizer, item, variant) for item in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        )
        input_device = next(model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        prompt_width = inputs["input_ids"].shape[1]
        for item, output in zip(batch, outputs):
            new_tokens = output[prompt_width:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            generated_tokens = int(new_tokens.numel())
            records.append(
                {
                    "id": int(item["id"]),
                    "is_mcq": is_mcq(item),
                    "response": response,
                    "answer_key": extract_answer_key(item, response, strict=True),
                    "generated_tokens": generated_tokens,
                    "hit_token_limit": generated_tokens >= args.max_new_tokens,
                }
            )
    del model
    torch.cuda.empty_cache()
    return records


def run_finalizer(raw_records: list[dict[str, Any]], data_by_id: dict[int, dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    finalizer_args = SimpleNamespace(
        model_id=args.model_name,
        adapter_dir=args.answer_finalizer_adapter,
        max_trace_chars=args.finalizer_max_trace_chars,
        max_input_tokens=args.finalizer_max_input_tokens,
        max_new_tokens=args.finalizer_max_new_tokens,
        temperature=args.finalizer_temperature,
        top_p=args.finalizer_top_p,
        retry_bad_format=args.retry_bad_format,
        retry_max_new_tokens=args.retry_max_new_tokens,
    )
    return FINALIZE.finalize_with_transformers(raw_records, data_by_id, finalizer_args)


def attach_metrics(items: list[dict[str, Any]], records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_by_id = index_by_id(items)
    judger = Judger(strict_extract=False)
    parse_failures = 0
    slot_mismatches = 0
    truncations = 0
    scored: list[dict[str, Any]] = []
    for record in records:
        item = data_by_id[int(record["id"])]
        response = str(record.get("response", ""))
        diagnostics = final_answer_diagnostics(item, response)
        strict_key = extract_answer_key(item, response, strict=True)
        if not strict_key:
            parse_failures += 1
        if not is_mcq(item) and diagnostics["parsed_slots"] != answer_slot_count(item):
            slot_mismatches += 1
        if record.get("hit_token_limit"):
            truncations += 1
        enriched = {**record, **diagnostics, "answer_key": strict_key}
        if has_gold(item):
            enriched["gold"] = item["answer"]
            enriched["correct"] = score_item(judger, item, response)
        scored.append(enriched)

    summary = summarize_results(scored)
    summary["diagnostics"] = {
        "parse_failures": parse_failures,
        "truncation_count": truncations,
        "slot_count_mismatch_count": slot_mismatches,
    }
    return scored, summary


def main() -> None:
    args = parse_args()
    items = read_items(args.data_file)
    if args.offset:
        items = items[args.offset :]
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise RuntimeError("No diagnostic rows to evaluate.")

    raw_records = generate_solver_outputs(items, args)
    write_jsonl(args.output_raw, raw_records)
    data_by_id = index_by_id(items)

    if args.without_finalizer:
        final_records = raw_records
    else:
        if not args.answer_finalizer_adapter:
            raise ValueError("--answer-finalizer-adapter is required unless --without-finalizer is set")
        final_records = run_finalizer(raw_records, data_by_id, args)

    scored, summary = attach_metrics(items, final_records)
    write_jsonl(args.output_final, scored)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote raw solver output to {args.output_raw}")
    print(f"Wrote evaluated output to {args.output_final}")


if __name__ == "__main__":
    main()
