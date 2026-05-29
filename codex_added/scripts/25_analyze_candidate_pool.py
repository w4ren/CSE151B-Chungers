# Added by Codex: candidate-pool diagnostics; not part of the original starter repository.

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
from math_comp.data import has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.final_answer import final_answer_diagnostics, normalize_final_response
from math_comp.scoring import extract_answer_key, score_item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze pass@budget, oracle lift, parse quality, and truncation.")
    parser.add_argument("--data", default="data/public.jsonl", help="Competition data JSONL.")
    parser.add_argument("--responses", nargs="+", required=True, help="Candidate JSONL files.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels matching --responses.")
    parser.add_argument("--output", default=None, help="Optional JSONL with per-problem candidate diagnostics.")
    parser.add_argument("--summary-output", default=None, help="Optional JSON summary path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit data rows after offset.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N data rows.")
    return parser.parse_args()


def label_for(path: str, index: int, labels: list[str] | None) -> str:
    if labels and index < len(labels):
        return labels[index]
    parent = Path(path).parent.name
    return parent if parent else Path(path).stem


def candidate_from_row(
    item: dict[str, Any],
    row: dict[str, Any],
    label: str,
    source_path: str,
    source_index: int,
    score_judger: Judger,
) -> dict[str, Any]:
    response = normalize_final_response(item, str(row.get("response", "")))
    answer_key = extract_answer_key(item, response, strict=True)
    diagnostics = final_answer_diagnostics(item, response)
    candidate = {
        "source": label,
        "source_path": source_path,
        "source_index": source_index,
        "response": response,
        "answer_key": answer_key,
        "generated_tokens": int(row.get("generated_tokens") or 0),
        "hit_token_limit": bool(row.get("hit_token_limit", False)),
        **diagnostics,
    }
    if has_gold(item):
        candidate["correct"] = score_item(score_judger, item, response)
    return candidate


def summarize_source(candidates: list[dict[str, Any]], with_gold: bool) -> dict[str, Any]:
    total = len(candidates)
    correct = sum(bool(candidate.get("correct")) for candidate in candidates)
    return {
        "total": total,
        "accuracy": correct / total if with_gold and total else None,
        "correct": correct if with_gold else None,
        "format_ok_rate": sum(bool(candidate.get("format_ok")) for candidate in candidates) / total if total else 0.0,
        "answer_key_rate": sum(bool(candidate.get("answer_key")) for candidate in candidates) / total if total else 0.0,
        "hit_token_limit_rate": sum(bool(candidate.get("hit_token_limit")) for candidate in candidates) / total if total else 0.0,
        "avg_generated_tokens": (
            sum(int(candidate.get("generated_tokens") or 0) for candidate in candidates) / total if total else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    if args.offset:
        data = data[args.offset :]
    if args.limit is not None:
        data = data[: args.limit]
    data_by_id = index_by_id(data)
    with_gold = all(has_gold(item) for item in data)
    score_judger = Judger(strict_extract=False)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels = args.labels if args.labels else None
    for source_index, path in enumerate(args.responses):
        label = label_for(path, source_index, labels)
        for row in read_jsonl(path):
            item = data_by_id.get(int(row["id"]))
            if item is None:
                continue
            candidate = candidate_from_row(item, row, label, path, source_index, score_judger)
            grouped[int(item["id"])].append(candidate)
            by_source[label].append(candidate)

    per_item_rows: list[dict[str, Any]] = []
    oracle_correct = 0
    disagreement = 0
    for item in data:
        item_id = int(item["id"])
        candidates = grouped.get(item_id, [])
        answer_keys = [candidate["answer_key"] for candidate in candidates if candidate.get("answer_key")]
        key_counts = Counter(answer_keys)
        if len(key_counts) > 1:
            disagreement += 1
        item_oracle = any(bool(candidate.get("correct")) for candidate in candidates) if has_gold(item) else None
        if item_oracle:
            oracle_correct += 1
        per_item_rows.append(
            {
                "id": item_id,
                "is_mcq": is_mcq(item),
                "num_candidates": len(candidates),
                "num_answer_keys": len(key_counts),
                "answer_key_counts": dict(key_counts),
                "oracle_correct": item_oracle,
                "candidates": candidates,
            }
        )

    summary = {
        "total_items": len(data),
        "with_gold": with_gold,
        "sources": {label: summarize_source(rows, with_gold) for label, rows in sorted(by_source.items())},
        "oracle_accuracy": oracle_correct / len(data) if with_gold and data else None,
        "oracle_correct": oracle_correct if with_gold else None,
        "disagreement_rate": disagreement / len(data) if data else 0.0,
        "disagreement_items": disagreement,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output:
        write_jsonl(args.output, per_item_rows)
    if args.summary_output:
        out_path = Path(args.summary_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
