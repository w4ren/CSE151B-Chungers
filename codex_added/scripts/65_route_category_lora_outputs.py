# Added by Codex: route category-specific LoRA outputs over a baseline prediction file.

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from judger import Judger
from math_comp.data import answer_slot_count, has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.final_answer import final_answer_diagnostics, normalize_final_response
from math_comp.grpo_finalizer import expected_answer_kind
from math_comp.scoring import extract_answer_key, score_item, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route category LoRA outputs on top of baseline predictions.")
    parser.add_argument("--data", default="data/public.jsonl")
    parser.add_argument("--base", required=True)
    parser.add_argument("--mcq-selector", default=None)
    parser.add_argument("--freeform-finalizer", default=None)
    parser.add_argument("--schema-finalizer", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score", action="store_true")
    parser.add_argument(
        "--freeform-kinds",
        default="multi_slot,decimal,exact_fraction,integer,yes_no,true_false,ordered_list,expression,unknown",
        help="Comma-separated expected_answer_kind values allowed to use freeform/schema outputs.",
    )
    return parser.parse_args()


def maybe_index(path: str | None) -> dict[int, dict[str, Any]]:
    return index_by_id(read_jsonl(path)) if path else {}


def source_for_item(item: dict[str, Any], args: argparse.Namespace, maps: dict[str, dict[int, dict[str, Any]]]) -> tuple[str, dict[str, Any] | None]:
    item_id = int(item["id"])
    if is_mcq(item) and item_id in maps["mcq"]:
        return "mcq_selector", maps["mcq"][item_id]
    allowed = {part.strip() for part in args.freeform_kinds.split(",") if part.strip()}
    kind = expected_answer_kind(item)
    if not is_mcq(item) and kind in allowed:
        if item_id in maps["schema"]:
            return "schema_finalizer", maps["schema"][item_id]
        if item_id in maps["freeform"]:
            return "freeform_finalizer", maps["freeform"][item_id]
    if item_id in maps["base"]:
        return "base", maps["base"][item_id]
    return "missing", None


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    maps = {
        "base": maybe_index(args.base),
        "mcq": maybe_index(args.mcq_selector),
        "freeform": maybe_index(args.freeform_finalizer),
        "schema": maybe_index(args.schema_finalizer),
    }
    judger = Judger(strict_extract=False)
    rows: list[dict[str, Any]] = []
    source_counts = Counter()
    for item in data:
        source, selected = source_for_item(item, args, maps)
        response = normalize_final_response(item, str((selected or {}).get("response", "")))
        row = {
            **(selected or {}),
            "id": int(item["id"]),
            "is_mcq": is_mcq(item),
            "response": response,
            "answer_key": extract_answer_key(item, response, strict=True),
            "route_source": source,
            "expected_answer_kind": expected_answer_kind(item),
            "num_ans_slots": answer_slot_count(item),
        }
        row.update(final_answer_diagnostics(item, response))
        if args.score and has_gold(item):
            row["gold"] = item["answer"]
            row["correct"] = score_item(judger, item, response)
        source_counts[source] += 1
        rows.append(row)
    write_jsonl(args.output, rows)
    summary = {
        "output": args.output,
        "source_counts": dict(sorted(source_counts.items())),
        "summary": summarize_results(rows) if args.score else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
