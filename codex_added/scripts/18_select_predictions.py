# Added by Codex: prediction policy combiner; not part of the original starter repository.

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

from judger import Judger
from math_comp.data import has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.scoring import extract_answer_key, score_item, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine two Qwen prediction JSONLs with a simple format-aware policy.")
    parser.add_argument("--data", default="data/public.jsonl", help="Competition data JSONL for ordering and scoring.")
    parser.add_argument("--base", required=True, help="Default prediction JSONL.")
    parser.add_argument("--override", required=True, help="Candidate override prediction JSONL.")
    parser.add_argument("--output", default="codex_added/results/selected.jsonl", help="Selected JSONL output.")
    parser.add_argument("--limit", type=int, default=None, help="Limit data rows after offset.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N data rows.")
    parser.add_argument(
        "--policy",
        choices=["free_form_override", "mcq_override", "all_override"],
        default="free_form_override",
        help="Which rows should prefer the override file.",
    )
    parser.add_argument("--score", action="store_true", help="Score selected output when data has answers.")
    return parser.parse_args()


def should_use_override(item: dict[str, Any], policy: str) -> bool:
    if policy == "all_override":
        return True
    if policy == "free_form_override":
        return not is_mcq(item)
    if policy == "mcq_override":
        return is_mcq(item)
    raise ValueError(f"Unsupported policy: {policy}")


def with_metadata(
    item: dict[str, Any],
    selected: dict[str, Any],
    source: str,
    score_judger: Judger,
    score: bool,
) -> dict[str, Any]:
    response = str(selected.get("response", ""))
    row = {
        **selected,
        "id": int(item["id"]),
        "is_mcq": is_mcq(item),
        "response": response,
        "answer_key": extract_answer_key(item, response, strict=True),
        "selected_source": source,
    }
    if score and has_gold(item):
        row["gold"] = item["answer"]
        row["correct"] = score_item(score_judger, item, response)
    return row


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    if args.offset:
        data = data[args.offset :]
    if args.limit is not None:
        data = data[: args.limit]
    base_by_id = index_by_id(read_jsonl(args.base))
    override_by_id = index_by_id(read_jsonl(args.override))
    score_judger = Judger(strict_extract=False)

    selected_rows: list[dict[str, Any]] = []
    for item in data:
        item_id = int(item["id"])
        base = base_by_id.get(item_id)
        override = override_by_id.get(item_id)
        if base is None and override is None:
            selected = {"id": item_id, "response": ""}
            source = "missing"
        elif override is not None and should_use_override(item, args.policy):
            selected = override
            source = "override"
        else:
            selected = base if base is not None else override
            source = "base" if base is not None else "override_fallback"
        selected_rows.append(with_metadata(item, selected, source, score_judger, args.score))

    write_jsonl(args.output, selected_rows)
    print(f"Wrote selected predictions to {args.output}")
    if args.score:
        print(json.dumps(summarize_results(selected_rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
