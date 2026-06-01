# Added by Codex: build locked-split GRPO finalizer data from public traces.

from __future__ import annotations

import argparse
import json
import random
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
from math_comp.grpo_finalizer import build_finalizer_messages, format_profile
from math_comp.scoring import extract_answer_key, score_item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GRPO finalizer train/holdout JSONL from labeled public traces.")
    parser.add_argument("--public", default="data/public.jsonl", help="Labeled public JSONL.")
    parser.add_argument("--raw-responses", required=True, help="Raw solver trace JSONL, usually full-public raw 8k.")
    parser.add_argument("--current-finalizer", default=None, help="Optional current finalizer JSONL for context/audits.")
    parser.add_argument("--output-dir", default="codex_added/data/grpo_finalizer")
    parser.add_argument("--train-output", default=None)
    parser.add_argument("--holdout-output", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--holdout-ids-file", default=None, help="Existing locked holdout ids. If absent, one is created.")
    parser.add_argument("--write-holdout-ids", default=None, help="Path to write generated holdout ids.")
    parser.add_argument("--holdout-count", type=int, default=200)
    parser.add_argument("--holdout-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1519)
    parser.add_argument("--max-trace-chars", type=int, default=6000)
    parser.add_argument("--max-traces-per-id", type=int, default=4)
    parser.add_argument("--include-correct-raw-only", action="store_true")
    return parser.parse_args()


def read_id_file(path: str | None) -> set[int]:
    if not path:
        return set()
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return set()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("ids", "holdout_ids", "ordered_selected_ids"):
                if key in data:
                    return {int(value) for value in data[key]}
        if isinstance(data, list):
            return {int(value) for value in data}
    except json.JSONDecodeError:
        pass
    return {int(line.strip()) for line in text.splitlines() if line.strip()}


def stratum(item: dict[str, Any]) -> str:
    if is_mcq(item):
        return "mcq"
    slots = answer_slot_count(item)
    if slots <= 1:
        return "free_single"
    if slots <= 4:
        return "free_multi_small"
    return "free_multi_large"


def make_holdout_ids(items: list[dict[str, Any]], args: argparse.Namespace) -> set[int]:
    supplied = read_id_file(args.holdout_ids_file)
    if supplied:
        return supplied

    rng = random.Random(args.seed)
    grouped: dict[str, list[int]] = defaultdict(list)
    for item in items:
        grouped[stratum(item)].append(int(item["id"]))

    if args.holdout_fraction > 0:
        total_target = max(1, round(len(items) * args.holdout_fraction))
    else:
        total_target = min(args.holdout_count, len(items))

    selected: set[int] = set()
    remaining_target = total_target
    strata = sorted(grouped)
    for name in strata:
        ids = grouped[name]
        rng.shuffle(ids)
        take = min(len(ids), max(1, round(total_target * len(ids) / len(items))))
        selected.update(ids[:take])
        remaining_target -= take

    if remaining_target > 0:
        leftovers = [int(item["id"]) for item in items if int(item["id"]) not in selected]
        rng.shuffle(leftovers)
        selected.update(leftovers[:remaining_target])
    if len(selected) > total_target:
        selected = set(sorted(selected)[:total_target])
    return selected


def first_by_id(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out.setdefault(int(row["id"]), row)
    return out


def score_or_none(judger: Judger, item: dict[str, Any], response: str) -> bool | None:
    if not has_gold(item):
        return None
    try:
        return score_item(judger, item, response)
    except Exception:
        return False


def build_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    public = [row for row in read_jsonl(args.public) if has_gold(row)]
    data_by_id = index_by_id(public)
    holdout_ids = make_holdout_ids(public, args)
    raw_rows = read_jsonl(args.raw_responses)
    current_by_id = first_by_id(read_jsonl(args.current_finalizer)) if args.current_finalizer else {}
    judger = Judger(strict_extract=False)

    trace_counts: Counter[int] = Counter()
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    skipped = Counter()

    for raw_index, raw in enumerate(raw_rows):
        item_id = int(raw["id"])
        item = data_by_id.get(item_id)
        if item is None:
            skipped["missing_public_item"] += 1
            continue
        if trace_counts[item_id] >= args.max_traces_per_id:
            skipped["max_traces_per_id"] += 1
            continue
        trace_counts[item_id] += 1

        solver_trace = str(raw.get("response", ""))
        raw_extracted = str(raw.get("answer_key") or extract_answer_key(item, solver_trace, strict=True))
        raw_correct = score_or_none(judger, item, solver_trace)
        if args.include_correct_raw_only and raw_correct is not True:
            skipped["raw_not_correct"] += 1
            continue

        current = current_by_id.get(item_id, {})
        current_answer = str(current.get("response", ""))
        current_correct = score_or_none(judger, item, current_answer) if current_answer else None
        split = "holdout" if item_id in holdout_ids else "train"
        profile = format_profile(item)
        record = {
            "example_id": f"{split}-{item_id}-{raw_index}",
            "id": item_id,
            "split": split,
            "problem": item["question"],
            "options": item.get("options") or [],
            "answer": item["answer"],
            "solver_trace": solver_trace,
            "raw_extracted_answer": raw_extracted,
            "raw_correct": raw_correct,
            "current_finalizer_answer": current_answer,
            "current_finalizer_correct": current_correct,
            "format_profile": profile,
            "messages": build_finalizer_messages(
                item,
                solver_trace,
                raw_extracted_answer=raw_extracted,
                current_finalizer_answer=current_answer,
                max_trace_chars=args.max_trace_chars,
            ),
            "source_raw_responses": args.raw_responses,
            "source_raw_row_index": raw_index,
        }
        if split == "holdout":
            holdout.append(record)
        else:
            train.append(record)

    manifest = {
        "public": args.public,
        "raw_responses": args.raw_responses,
        "current_finalizer": args.current_finalizer,
        "seed": args.seed,
        "max_trace_chars": args.max_trace_chars,
        "max_traces_per_id": args.max_traces_per_id,
        "holdout_ids_count": len(holdout_ids),
        "holdout_ids": sorted(holdout_ids),
        "train_examples": len(train),
        "holdout_examples": len(holdout),
        "skipped": dict(sorted(skipped.items())),
        "train_by_kind": dict(Counter(row["format_profile"]["expected_answer_kind"] for row in train)),
        "holdout_by_kind": dict(Counter(row["format_profile"]["expected_answer_kind"] for row in holdout)),
    }
    return train, holdout, manifest


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    train_path = Path(args.train_output) if args.train_output else out_dir / "train.jsonl"
    holdout_path = Path(args.holdout_output) if args.holdout_output else out_dir / "holdout.jsonl"
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.json"
    holdout_ids_path = Path(args.write_holdout_ids) if args.write_holdout_ids else out_dir / "locked_holdout_ids.txt"

    train, holdout, manifest = build_examples(args)
    write_jsonl(train_path, train)
    write_jsonl(holdout_path, holdout)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    holdout_ids_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_ids_path.write_text("\n".join(str(item_id) for item_id in manifest["holdout_ids"]) + "\n", encoding="utf-8")
    print(f"Wrote {len(train)} train examples to {train_path}")
    print(f"Wrote {len(holdout)} holdout examples to {holdout_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
