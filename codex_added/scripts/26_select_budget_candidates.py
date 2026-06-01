# Added by Codex: budget-sweep candidate selector; not part of the original starter repository.

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
from math_comp.scoring import extract_answer_key, score_item, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select one answer per problem from 4k/8k/16k-style budget sweeps.")
    parser.add_argument("--data", default="data/public.jsonl", help="Competition data JSONL.")
    parser.add_argument("--responses", nargs="+", required=True, help="Candidate JSONL files in increasing budget order.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels matching --responses.")
    parser.add_argument("--output", default="codex_added/results/budget_selected.jsonl", help="Selected JSONL output.")
    parser.add_argument("--limit", type=int, default=None, help="Limit data rows after offset.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N data rows.")
    parser.add_argument("--score", action="store_true", help="Score selected output when data has answers.")
    return parser.parse_args()


def label_for(path: str, index: int, labels: list[str] | None) -> str:
    if labels and index < len(labels):
        return labels[index]
    parent = Path(path).parent.name
    return parent if parent else Path(path).stem


def read_candidates(
    paths: list[str],
    labels: list[str] | None,
    data_by_id: dict[int, dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source_index, path in enumerate(paths):
        label = label_for(path, source_index, labels)
        for row in read_jsonl(path):
            item = data_by_id.get(int(row["id"]))
            if item is None:
                continue
            response = normalize_final_response(item, str(row.get("response", "")))
            candidate = {
                **row,
                "id": int(row["id"]),
                "source": label,
                "source_index": source_index,
                "source_path": path,
                "response": response,
                "answer_key": extract_answer_key(item, response, strict=True),
                "generated_tokens": int(row.get("generated_tokens") or 0),
                "hit_token_limit": bool(row.get("hit_token_limit", False)),
            }
            candidate.update(final_answer_diagnostics(item, response))
            grouped[int(row["id"])].append(candidate)
    return grouped


def candidate_rank(candidate: dict[str, Any], key_vote_count: int) -> tuple[int, int, int, int, int]:
    return (
        key_vote_count,
        1 if candidate.get("format_ok") else 0,
        0 if candidate.get("hit_token_limit") else 1,
        int(candidate.get("source_index", 0)),
        int(candidate.get("generated_tokens") or 0),
    )


def choose_candidate(item: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    if not candidates:
        return {"id": int(item["id"]), "response": "", "answer_key": ""}, "missing"

    answer_keys = [str(candidate.get("answer_key", "")) for candidate in candidates if candidate.get("answer_key")]
    key_counts = Counter(answer_keys)
    if key_counts:
        best_count = max(key_counts.values())
        winning_keys = {key for key, count in key_counts.items() if count == best_count}
        keyed = [candidate for candidate in candidates if str(candidate.get("answer_key", "")) in winning_keys]
        selected = max(keyed, key=lambda candidate: candidate_rank(candidate, key_counts[str(candidate.get("answer_key", ""))]))
        policy = "answer_key_consensus" if best_count > 1 else "best_single_key"
    else:
        selected = max(candidates, key=lambda candidate: candidate_rank(candidate, 0))
        policy = "no_parse_fallback"

    if not is_mcq(item) and not selected.get("format_ok"):
        formatted = [candidate for candidate in candidates if candidate.get("format_ok")]
        if formatted:
            selected = max(formatted, key=lambda candidate: candidate_rank(candidate, key_counts[str(candidate.get("answer_key", ""))]))
            policy = f"{policy}_format_repair"

    return selected, policy


def with_selection_metadata(
    item: dict[str, Any],
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
    policy: str,
    score_judger: Judger,
    score: bool,
) -> dict[str, Any]:
    response = normalize_final_response(item, str(selected.get("response", "")))
    row = {
        **selected,
        "id": int(item["id"]),
        "is_mcq": is_mcq(item),
        "response": response,
        "answer_key": extract_answer_key(item, response, strict=True),
        "selected_source": selected.get("source", "missing"),
        "selection_policy": policy,
        "num_candidates": len(candidates),
        "candidate_answer_keys": [str(candidate.get("answer_key", "")) for candidate in candidates],
    }
    row.update(final_answer_diagnostics(item, response))
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

    data_by_id = index_by_id(data)
    grouped = read_candidates(args.responses, args.labels, data_by_id)
    score_judger = Judger(strict_extract=False)

    selected_rows: list[dict[str, Any]] = []
    for item in data:
        candidates = grouped.get(int(item["id"]), [])
        selected, policy = choose_candidate(item, candidates)
        selected_rows.append(with_selection_metadata(item, selected, candidates, policy, score_judger, args.score))

    write_jsonl(args.output, selected_rows)
    print(f"Wrote selected predictions to {args.output}")
    if args.score:
        print(json.dumps(summarize_results(selected_rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
