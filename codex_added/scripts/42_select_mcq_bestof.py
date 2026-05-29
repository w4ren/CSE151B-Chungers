# Added by Codex: MCQ-only best-of-N selection and oracle diagnostics.

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
    parser = argparse.ArgumentParser(description="Select MCQ answers from a best-of-N candidate pool.")
    parser.add_argument("--data", required=True, help="MCQ-only data JSONL.")
    parser.add_argument("--responses", nargs="+", required=True, help="Candidate JSONL files.")
    parser.add_argument("--baseline", default=None, help="Optional current-best baseline JSONL for fallback/tie-break.")
    parser.add_argument("--output", required=True, help="Selected JSONL for the main policy.")
    parser.add_argument("--summary-output", default=None, help="Optional JSON summary path.")
    parser.add_argument("--details-output", default=None, help="Optional JSONL diagnostics path.")
    parser.add_argument(
        "--policy",
        choices=["candidate_majority", "consensus_or_baseline", "baseline_plus_vote"],
        default="consensus_or_baseline",
        help="Selection policy to write to --output.",
    )
    parser.add_argument(
        "--min-consensus",
        type=int,
        default=2,
        help="Minimum top vote count before replacing baseline for consensus_or_baseline.",
    )
    parser.add_argument(
        "--baseline-weight",
        type=int,
        default=1,
        help="Vote weight assigned to baseline in baseline_plus_vote.",
    )
    parser.add_argument("--score", action="store_true", help="Score outputs when gold answers are present.")
    return parser.parse_args()


def label_for(path: str, index: int) -> str:
    parent = Path(path).parent.name
    return parent if parent else f"source_{index}"


def row_answer_key(item: dict[str, Any], row: dict[str, Any], judger: Judger) -> str:
    key = str(row.get("answer_key") or "").strip().upper()
    if len(key) == 1:
        return key
    return extract_answer_key(item, str(row.get("response", "")), judger=judger, strict=True)


def read_candidates(
    paths: list[str],
    data_by_id: dict[int, dict[str, Any]],
    key_judger: Judger,
    score_judger: Judger,
    score: bool,
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    order = 0
    for source_index, path in enumerate(paths):
        label = label_for(path, source_index)
        for source_row_index, row in enumerate(read_jsonl(path)):
            item = data_by_id.get(int(row["id"]))
            if item is None:
                continue
            if not is_mcq(item):
                continue
            response = str(row.get("response", ""))
            candidate = {
                **row,
                "id": int(row["id"]),
                "source": label,
                "source_path": path,
                "source_index": source_index,
                "source_row_index": source_row_index,
                "candidate_order": order,
                "response": response,
                "answer_key": row_answer_key(item, row, key_judger),
            }
            if score and has_gold(item):
                candidate["correct"] = score_item(score_judger, item, response)
            grouped[int(row["id"])].append(candidate)
            order += 1
    return grouped


def read_baseline(path: str | None, data_by_id: dict[int, dict[str, Any]], key_judger: Judger) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    baseline: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        item = data_by_id.get(int(row["id"]))
        if item is None or not is_mcq(item):
            continue
        response = str(row.get("response", ""))
        baseline[int(row["id"])] = {
            **row,
            "id": int(row["id"]),
            "source": "baseline",
            "response": response,
            "answer_key": row_answer_key(item, row, key_judger),
        }
    return baseline


def first_candidate_with_key(candidates: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    keyed = [candidate for candidate in candidates if str(candidate.get("answer_key", "")) == key]
    if not keyed:
        return None
    return min(keyed, key=lambda candidate: int(candidate.get("candidate_order") or 0))


def candidate_majority(
    item: dict[str, Any],
    candidates: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    min_consensus: int,
    baseline_weight: int,
) -> tuple[dict[str, Any], str, dict[str, int]]:
    keys = [str(candidate.get("answer_key", "")) for candidate in candidates if candidate.get("answer_key")]
    counts = Counter(keys)
    if not counts:
        return baseline or {"id": int(item["id"]), "response": "", "answer_key": ""}, "no_candidate_key", dict(counts)
    first_seen: dict[str, int] = {}
    for candidate in candidates:
        key = str(candidate.get("answer_key", ""))
        if key:
            first_seen.setdefault(key, int(candidate.get("candidate_order") or 0))
    winning_key = min(counts, key=lambda key: (-counts[key], first_seen[key]))
    return first_candidate_with_key(candidates, winning_key) or candidates[0], "candidate_majority", dict(counts)


def consensus_or_baseline(
    item: dict[str, Any],
    candidates: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    min_consensus: int,
    baseline_weight: int,
) -> tuple[dict[str, Any], str, dict[str, int]]:
    selected, _, counts = candidate_majority(item, candidates, baseline, min_consensus, baseline_weight)
    if not counts:
        return selected, "baseline_no_candidate_key" if baseline else "missing", counts
    top = max(counts.values())
    tied_keys = {key for key, count in counts.items() if count == top}
    baseline_key = str((baseline or {}).get("answer_key") or "")
    if baseline and (top < min_consensus or len(tied_keys) > 1):
        if baseline_key in tied_keys:
            return baseline, "baseline_tiebreak", counts
        return baseline, "baseline_low_consensus", counts
    return selected, "candidate_consensus", counts


def baseline_plus_vote(
    item: dict[str, Any],
    candidates: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    min_consensus: int,
    baseline_weight: int,
) -> tuple[dict[str, Any], str, dict[str, int]]:
    keys = [str(candidate.get("answer_key", "")) for candidate in candidates if candidate.get("answer_key")]
    counts = Counter(keys)
    baseline_key = str((baseline or {}).get("answer_key") or "")
    if baseline_key:
        counts[baseline_key] += max(1, baseline_weight)
    if not counts:
        return baseline or {"id": int(item["id"]), "response": "", "answer_key": ""}, "missing", dict(counts)
    first_seen: dict[str, int] = {}
    if baseline_key:
        first_seen[baseline_key] = -1
    for candidate in candidates:
        key = str(candidate.get("answer_key", ""))
        if key:
            first_seen.setdefault(key, int(candidate.get("candidate_order") or 0))
    winning_key = min(counts, key=lambda key: (-counts[key], first_seen.get(key, 10**9)))
    if baseline and winning_key == baseline_key:
        return baseline, "baseline_plus_vote", dict(counts)
    selected = first_candidate_with_key(candidates, winning_key)
    return selected or baseline or candidates[0], "candidate_plus_vote", dict(counts)


POLICIES = {
    "candidate_majority": candidate_majority,
    "consensus_or_baseline": consensus_or_baseline,
    "baseline_plus_vote": baseline_plus_vote,
}


def selected_row(
    item: dict[str, Any],
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
    policy: str,
    reason: str,
    counts: dict[str, int],
    score_judger: Judger,
    score: bool,
) -> dict[str, Any]:
    response = str(selected.get("response", ""))
    row = {
        "id": int(item["id"]),
        "is_mcq": True,
        "response": response,
        "answer_key": extract_answer_key(item, response, strict=True),
        "selected_source": selected.get("source", "missing"),
        "selected_candidate_order": selected.get("candidate_order"),
        "selection_policy": policy,
        "selection_reason": reason,
        "num_candidates": len(candidates),
        "candidate_answer_counts": counts,
        "candidate_answer_keys": [str(candidate.get("answer_key", "")) for candidate in candidates],
    }
    if score and has_gold(item):
        row["gold"] = item["answer"]
        row["correct"] = score_item(score_judger, item, response)
    return row


def summarize_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_results(rows)
    reasons = Counter(str(row.get("selection_reason", "")) for row in rows)
    summary["selection_reasons"] = dict(sorted(reasons.items()))
    return summary


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    if any(not is_mcq(item) for item in data):
        raise ValueError("--data must be MCQ-only")
    data_by_id = index_by_id(data)
    key_judger = Judger(strict_extract=True)
    score_judger = Judger(strict_extract=False)
    grouped = read_candidates(args.responses, data_by_id, key_judger, score_judger, args.score)
    baseline_by_id = read_baseline(args.baseline, data_by_id, key_judger)

    policy_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in POLICIES}
    details: list[dict[str, Any]] = []
    oracle_correct = 0
    baseline_rows: list[dict[str, Any]] = []

    for item in data:
        item_id = int(item["id"])
        candidates = grouped.get(item_id, [])
        baseline = baseline_by_id.get(item_id)
        if args.score and has_gold(item):
            if any(bool(candidate.get("correct")) for candidate in candidates):
                oracle_correct += 1
            if baseline:
                baseline_rows.append(
                    {
                        "id": item_id,
                        "is_mcq": True,
                        "response": baseline["response"],
                        "correct": score_item(score_judger, item, str(baseline.get("response", ""))),
                    }
                )
        for policy_name, choose in POLICIES.items():
            selected, reason, counts = choose(
                item,
                candidates,
                baseline,
                args.min_consensus,
                args.baseline_weight,
            )
            policy_rows[policy_name].append(
                selected_row(item, selected, candidates, policy_name, reason, counts, score_judger, args.score)
            )
        details.append(
            {
                "id": item_id,
                "gold": item.get("answer"),
                "baseline_key": str((baseline or {}).get("answer_key", "")),
                "baseline_correct": (
                    score_item(score_judger, item, str((baseline or {}).get("response", "")))
                    if args.score and baseline and has_gold(item)
                    else None
                ),
                "oracle_correct": any(bool(candidate.get("correct")) for candidate in candidates)
                if args.score and has_gold(item)
                else None,
                "candidate_answer_keys": [str(candidate.get("answer_key", "")) for candidate in candidates],
                "candidates": candidates,
            }
        )

    selected = policy_rows[args.policy]
    write_jsonl(args.output, selected)
    if args.details_output:
        write_jsonl(args.details_output, details)

    summary = {
        "total_items": len(data),
        "num_candidate_rows": sum(len(rows) for rows in grouped.values()),
        "avg_candidates_per_item": (
            sum(len(rows) for rows in grouped.values()) / len(data) if data else 0.0
        ),
        "oracle": {
            "correct": oracle_correct if args.score else None,
            "total": len(data),
            "accuracy": oracle_correct / len(data) if args.score and data else None,
        },
        "baseline": summarize_results(baseline_rows) if baseline_rows else None,
        "policies": {name: summarize_policy(rows) for name, rows in policy_rows.items()},
        "written_policy": args.policy,
        "written_output": args.output,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_output:
        out_path = Path(args.summary_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
