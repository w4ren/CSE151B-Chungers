# Added by Codex: self-consistency voting; not part of the original starter repository.

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
from math_comp.scoring import extract_answer_key, score_item, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vote prompt-sweep traces by normalized final answer.")
    parser.add_argument("--data", default=None, help="Optional data JSONL for output ordering and scoring.")
    parser.add_argument("--responses", required=True, help="Sweep JSONL from 10_run_prompt_sweep.py.")
    parser.add_argument("--output", default="codex_added/results/voted_public.jsonl", help="Voted JSONL output.")
    parser.add_argument("--score", action="store_true", help="Score voted responses when --data has answers.")
    parser.add_argument("--include-empty", action="store_true", help="Allow empty answer keys to win votes.")
    parser.add_argument("--include-missing", action="store_true", help="Emit empty rows for data ids missing from responses.")
    parser.add_argument("--keep-answer-keys", action="store_true", help="Use answer_key fields already present in responses.")
    parser.add_argument(
        "--answer-key-mode",
        choices=["strict", "loose"],
        default="strict",
        help="How to recompute answer keys when --data is provided.",
    )
    return parser.parse_args()


def choose_winner(records: list[dict[str, Any]], include_empty: bool) -> dict[str, Any]:
    candidates = [
        (idx, record, str(record.get("answer_key", "")))
        for idx, record in enumerate(records)
        if include_empty or str(record.get("answer_key", ""))
    ]
    if not candidates:
        first = records[0]
        return {
            "answer_key": "",
            "vote_count": 0,
            "response": str(first.get("response", "")),
            "variant": first.get("variant"),
            "sample_index": first.get("sample_index"),
            "candidate_counts": [],
        }

    counts = Counter(key for _, _, key in candidates)
    first_seen: dict[str, int] = {}
    for idx, _, key in candidates:
        first_seen.setdefault(key, idx)

    winning_key = min(counts, key=lambda key: (-counts[key], first_seen[key]))
    winner_record = next(record for _, record, key in candidates if key == winning_key)
    candidate_counts = [
        {"answer_key": key, "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    ]

    return {
        "answer_key": winning_key,
        "vote_count": counts[winning_key],
        "response": str(winner_record.get("response", "")),
        "variant": winner_record.get("variant"),
        "sample_index": winner_record.get("sample_index"),
        "candidate_counts": candidate_counts,
    }


def main() -> None:
    args = parse_args()
    sweep_records = read_jsonl(args.responses)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in sweep_records:
        grouped[int(record["id"])].append(record)

    data = read_jsonl(args.data) if args.data else []
    data_by_id = index_by_id(data) if data else {}
    if data and args.include_missing:
        ordered_ids = [int(item["id"]) for item in data]
    elif data:
        data_order = [int(item["id"]) for item in data]
        ordered_ids = [item_id for item_id in data_order if item_id in grouped]
    else:
        ordered_ids = sorted(grouped)
    key_judger = Judger(strict_extract=args.answer_key_mode == "strict")
    score_judger = Judger(strict_extract=False)

    voted: list[dict[str, Any]] = []
    for item_id in ordered_ids:
        records = grouped.get(item_id, [])
        if not records:
            voted.append({"id": item_id, "is_mcq": bool(data_by_id.get(item_id, {}).get("options")), "response": ""})
            continue

        item = data_by_id.get(item_id)
        if item:
            for record in records:
                if not args.keep_answer_keys or not record.get("answer_key"):
                    record["answer_key"] = extract_answer_key(
                        item,
                        str(record.get("response", "")),
                        key_judger,
                        strict=args.answer_key_mode == "strict",
                    )

        winner = choose_winner(records, args.include_empty)
        row = {
            "id": item_id,
            "is_mcq": is_mcq(item) if item else bool(records[0].get("is_mcq")),
            "answer_key": winner["answer_key"],
            "vote_count": winner["vote_count"],
            "num_traces": len(records),
            "variant": winner["variant"],
            "sample_index": winner["sample_index"],
            "candidate_counts": winner["candidate_counts"],
            "response": winner["response"],
        }

        if args.score and item and has_gold(item):
            row["gold"] = item["answer"]
            row["correct"] = score_item(score_judger, item, row["response"])
        voted.append(row)

    write_jsonl(args.output, voted)
    print(f"Wrote voted responses to {args.output}")
    if args.score:
        print(json.dumps(summarize_results(voted), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
