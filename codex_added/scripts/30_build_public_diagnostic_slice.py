# Added by Codex: build a held-out diagnostic slice for formatting/repair audits.

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

from math_comp.data import answer_slot_count, is_mcq, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a separate 100-row public diagnostic slice.")
    parser.add_argument("--data", default="data/public.jsonl", help="Source public JSONL.")
    parser.add_argument(
        "--output",
        default="codex_added/job_data/public_diagnostic_100_offset200_40mcq_40multislot_20hard.jsonl",
        help="Output diagnostic JSONL.",
    )
    parser.add_argument(
        "--ids-output",
        default="codex_added/job_data/public_diagnostic_100_offset200_40mcq_40multislot_20hard_ids.txt",
        help="Output text file containing selected ids.",
    )
    parser.add_argument(
        "--manifest-output",
        default="codex_added/job_data/public_diagnostic_100_offset200_40mcq_40multislot_20hard_manifest.json",
        help="Output manifest JSON.",
    )
    parser.add_argument("--min-id", type=int, default=200, help="Only use rows with id >= this value.")
    parser.add_argument("--mcq-count", type=int, default=40)
    parser.add_argument("--multislot-count", type=int, default=40)
    parser.add_argument("--hard-count", type=int, default=20)
    return parser.parse_args()


def hard_reasoning_score(item: dict[str, Any]) -> tuple[int, int, int]:
    question = str(item.get("question", ""))
    math_markers = sum(
        question.count(marker)
        for marker in (
            "\\frac",
            "frac",
            "\\sum",
            "sum",
            "\\int",
            "int_",
            "\\prod",
            "\\sqrt",
            "sqrt",
            "\\log",
            "ln",
            "derivative",
            "probability",
            "expected",
            "prove",
            "sequence",
            "matrix",
        )
    )
    # Longer single-slot free-form rows with more math syntax are a decent
    # deterministic proxy for "hard reasoning" before model traces exist.
    return (math_markers, len(question), int(item["id"]))


def take_or_fail(label: str, rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"Need {count} {label} rows, but only found {len(rows)}")
    return rows[:count]


def interleave(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    max_len = max(len(group) for group in groups)
    for index in range(max_len):
        for group in groups:
            if index < len(group):
                selected.append(group[index])
    return selected


def main() -> None:
    args = parse_args()
    items = [item for item in read_jsonl(args.data) if int(item["id"]) >= args.min_id]

    mcq_rows = [item for item in items if is_mcq(item)]
    multislot_rows = [item for item in items if not is_mcq(item) and answer_slot_count(item) >= 2]
    hard_rows = [item for item in items if not is_mcq(item) and answer_slot_count(item) == 1]
    hard_rows.sort(key=hard_reasoning_score, reverse=True)

    selected_mcq = [
        {**item, "diagnostic_group": "mcq", "diagnostic_reason": "multiple_choice"}
        for item in take_or_fail("MCQ", mcq_rows, args.mcq_count)
    ]
    selected_multislot = [
        {
            **item,
            "diagnostic_group": "multi_slot_freeform",
            "diagnostic_reason": f"{answer_slot_count(item)} answer slots",
        }
        for item in take_or_fail("multi-slot free-form", multislot_rows, args.multislot_count)
    ]
    selected_hard = [
        {
            **item,
            "diagnostic_group": "hard_reasoning",
            "diagnostic_reason": "single-slot free-form selected by length/math-marker heuristic",
        }
        for item in take_or_fail("hard reasoning", hard_rows, args.hard_count)
    ]

    selected = interleave([selected_mcq, selected_multislot, selected_hard])
    write_jsonl(args.output, selected)

    ids = [int(item["id"]) for item in selected]
    ids_path = Path(args.ids_output)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("\n".join(str(item_id) for item_id in ids) + "\n", encoding="utf-8")

    manifest = {
        "source": args.data,
        "output": args.output,
        "min_id": args.min_id,
        "total": len(selected),
        "counts": {
            "mcq": len(selected_mcq),
            "multi_slot_freeform": len(selected_multislot),
            "hard_reasoning": len(selected_hard),
        },
        "ids": ids,
    }
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
