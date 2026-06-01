# Added by Codex: compare supervised and GRPO finalizer outputs by category.

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from judger import Judger
from math_comp.data import answer_slot_count, has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.grpo_finalizer import expected_answer_kind
from math_comp.scoring import extract_answer_key, score_item, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline finalizer vs candidate GRPO finalizer.")
    parser.add_argument("--data", default="data/public.jsonl")
    parser.add_argument("--baseline", required=True, help="Supervised/current finalizer JSONL.")
    parser.add_argument("--candidate", required=True, help="GRPO finalizer JSONL.")
    parser.add_argument("--raw-responses", default=None, help="Optional raw trace JSONL for raw-correct and flip audits.")
    parser.add_argument("--holdout-ids-file", default=None)
    parser.add_argument("--output-json", default="codex_added/results/grpo_finalizer_compare_summary.json")
    parser.add_argument("--details-output", default="codex_added/results/grpo_finalizer_compare_details.jsonl")
    return parser.parse_args()


def read_ids(path: str | None) -> set[int]:
    if not path:
        return set()
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return set()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return {int(value) for value in data}
        if isinstance(data, dict):
            for key in ("ids", "holdout_ids", "ordered_selected_ids"):
                if key in data:
                    return {int(value) for value in data[key]}
    except json.JSONDecodeError:
        pass
    return {int(line.strip()) for line in text.splitlines() if line.strip()}


def category(item: dict[str, Any]) -> str:
    if is_mcq(item):
        return "mcq"
    slots = answer_slot_count(item)
    if slots <= 1:
        return f"free_single:{expected_answer_kind(item)}"
    return f"free_multi_{slots}_slots"


def score_row(judger: Judger, item: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    response = str((row or {}).get("response", ""))
    correct = score_item(judger, item, response) if has_gold(item) else None
    return {
        "response": response,
        "answer_key": extract_answer_key(item, response, strict=True),
        "correct": correct,
    }


def summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_correct = sum(row["baseline_correct"] is True for row in rows)
    candidate_correct = sum(row["candidate_correct"] is True for row in rows)
    gains = sum(row["baseline_correct"] is False and row["candidate_correct"] is True for row in rows)
    losses = sum(row["baseline_correct"] is True and row["candidate_correct"] is False for row in rows)
    total = len(rows)
    return {
        "total": total,
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "baseline_accuracy": baseline_correct / total if total else 0.0,
        "candidate_accuracy": candidate_correct / total if total else 0.0,
        "delta_correct": candidate_correct - baseline_correct,
        "gains": gains,
        "losses": losses,
    }


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    holdout_ids = read_ids(args.holdout_ids_file)
    if holdout_ids:
        data = [item for item in data if int(item["id"]) in holdout_ids]

    baseline_by_id = index_by_id(read_jsonl(args.baseline))
    candidate_by_id = index_by_id(read_jsonl(args.candidate))
    raw_by_id = index_by_id(read_jsonl(args.raw_responses)) if args.raw_responses else {}
    judger = Judger(strict_extract=False)

    details: list[dict[str, Any]] = []
    scored_baseline: list[dict[str, Any]] = []
    scored_candidate: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    flip_cases: list[dict[str, Any]] = []

    for item in data:
        item_id = int(item["id"])
        baseline = score_row(judger, item, baseline_by_id.get(item_id))
        candidate = score_row(judger, item, candidate_by_id.get(item_id))
        raw = score_row(judger, item, raw_by_id.get(item_id)) if raw_by_id else {"correct": None, "answer_key": ""}
        row = {
            "id": item_id,
            "category": category(item),
            "is_mcq": is_mcq(item),
            "gold": item.get("answer"),
            "raw_answer_key": raw.get("answer_key", ""),
            "raw_correct": raw.get("correct"),
            "baseline_response": baseline["response"],
            "baseline_answer_key": baseline["answer_key"],
            "baseline_correct": baseline["correct"],
            "candidate_response": candidate["response"],
            "candidate_answer_key": candidate["answer_key"],
            "candidate_correct": candidate["correct"],
        }
        if row["raw_correct"] is True and row["baseline_correct"] is True and row["candidate_correct"] is False:
            flip_cases.append(row)
        details.append(row)
        buckets[row["category"]].append(row)
        scored_baseline.append({"id": item_id, "is_mcq": is_mcq(item), "correct": baseline["correct"], "response": baseline["response"]})
        scored_candidate.append({"id": item_id, "is_mcq": is_mcq(item), "correct": candidate["correct"], "response": candidate["response"]})

    summary = {
        "data": args.data,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "holdout_ids_file": args.holdout_ids_file,
        "num_rows": len(details),
        "baseline_summary": summarize_results(scored_baseline),
        "candidate_summary": summarize_results(scored_candidate),
        "overall_compare": summarize_bucket(details),
        "by_category": {name: summarize_bucket(rows) for name, rows in sorted(buckets.items())},
        "selection_rule": {
            "use_candidate_globally": summarize_bucket(details)["delta_correct"] > 0 and summarize_bucket(details)["losses"] == 0,
            "candidate_wins_categories": [
                name for name, rows in sorted(buckets.items()) if summarize_bucket(rows)["delta_correct"] > 0
            ],
            "candidate_loses_categories": [
                name for name, rows in sorted(buckets.items()) if summarize_bucket(rows)["delta_correct"] < 0
            ],
        },
        "raw_or_baseline_correct_to_candidate_loss_count": len(flip_cases),
        "reason_counts": dict(Counter(row["category"] for row in details)),
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(args.details_output, details)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
