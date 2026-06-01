# Added by Codex: build MCQ selector LoRA SFT data from public traces and candidates.

from __future__ import annotations

import argparse
import json
import random
import re
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
from math_comp.prompts import format_options
from math_comp.scoring import extract_answer_key, score_item


SYSTEM = (
    "You are an MCQ answer selector for a math competition. "
    "Use the problem, options, solver trace, and candidate answers to output exactly one uppercase option letter. "
    "Do not output option text. Do not explain."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MCQ selector SFT chat data.")
    parser.add_argument("--public", default="data/public.jsonl")
    parser.add_argument("--raw-responses", nargs="+", required=True, help="Raw Qwen trace JSONL files.")
    parser.add_argument("--candidate-responses", nargs="*", default=[], help="Optional finalized/candidate JSONL files.")
    parser.add_argument("--baseline", default=None, help="Optional current best/finalizer predictions for context.")
    parser.add_argument("--holdout-ids-file", default=None)
    parser.add_argument("--holdout-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1521)
    parser.add_argument("--max-trace-chars", type=int, default=5000)
    parser.add_argument("--max-traces-per-id", type=int, default=4)
    parser.add_argument("--permutations-per-example", type=int, default=0)
    parser.add_argument("--boxed-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default="codex_added/data/mcq_selector_lora")
    parser.add_argument("--train-output", default=None)
    parser.add_argument("--holdout-output", default=None)
    parser.add_argument("--manifest", default=None)
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


def make_holdout_ids(items: list[dict[str, Any]], args: argparse.Namespace) -> set[int]:
    supplied = read_ids(args.holdout_ids_file)
    if supplied:
        return supplied
    rng = random.Random(args.seed)
    ids = [int(item["id"]) for item in items if is_mcq(item) and has_gold(item)]
    rng.shuffle(ids)
    return set(ids[: min(args.holdout_count, len(ids))])


def compact_trace(trace: str, max_chars: int) -> str:
    trace = str(trace or "").strip()
    if len(trace) <= max_chars:
        return trace
    head = max_chars // 4
    tail = max_chars - head
    omitted = len(trace) - head - tail
    return trace[:head].rstrip() + f"\n\n[... omitted {omitted} characters ...]\n\n" + trace[-tail:].lstrip()


def boxed(letter: str) -> str:
    return "\\boxed{" + letter.strip().upper() + "}"


def target_text(letter: str, boxed_target: bool) -> str:
    letter = letter.strip().upper()
    return boxed(letter) if boxed_target else letter


def rows_by_id(paths: list[str]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for row in read_jsonl(path):
            grouped[int(row["id"])].append({**row, "source_path": path})
    return grouped


def first_by_id(path: str | None) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        out.setdefault(int(row["id"]), row)
    return out


def candidate_summary(item: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, row in enumerate(rows[:12], start=1):
        response = str(row.get("response", ""))
        key = str(row.get("answer_key") or extract_answer_key(item, response, strict=True))
        source = Path(str(row.get("source_path", "candidate"))).name
        lines.append(f"{index}. source={source}, extracted_letter={key or '<none>'}, response_tail={response[-220:]}")
    return "\n".join(lines) if lines else "<none>"


def build_messages(
    item: dict[str, Any],
    raw_row: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    max_trace_chars: int,
    permuted_note: str = "",
) -> list[dict[str, str]]:
    trace = compact_trace(str(raw_row.get("response", "")), max_trace_chars)
    raw_key = str(raw_row.get("answer_key") or extract_answer_key(item, trace, strict=True))
    baseline_response = str((baseline or {}).get("response", ""))
    baseline_key = str((baseline or {}).get("answer_key") or extract_answer_key(item, baseline_response, strict=True)) if baseline else ""
    user = (
        "Problem:\n"
        f"{item['question']}\n\n"
        "Options:\n"
        f"{format_options(item.get('options') or [])}\n\n"
        f"{permuted_note}"
        "Solver trace:\n"
        f"{trace}\n\n"
        "Raw extracted answer letter:\n"
        f"{raw_key or '<none>'}\n\n"
        "Current baseline/finalizer answer letter:\n"
        f"{baseline_key or '<none>'}\n\n"
        "Candidate answers:\n"
        f"{candidate_summary(item, candidate_rows)}\n\n"
        "Final answer letter only:"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def permute_item(item: dict[str, Any], target_letter: str, rng: random.Random) -> tuple[dict[str, Any], str, str]:
    options = list(item.get("options") or [])
    if not options:
        return item, target_letter, ""
    gold_index = ord(target_letter.strip().upper()) - 65
    if gold_index < 0 or gold_index >= len(options):
        return item, target_letter, ""
    perm = list(range(len(options)))
    rng.shuffle(perm)
    if perm == list(range(len(options))):
        perm = perm[1:] + perm[:1]
    new_options = [options[index] for index in perm]
    new_target = chr(65 + perm.index(gold_index))
    note = (
        "Note: option order was permuted for training. Solver trace letter mentions may refer to the original order. "
        "Select by matching the option content to the problem.\n\n"
    )
    return {**item, "options": new_options}, new_target, note


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    public = [item for item in read_jsonl(args.public) if is_mcq(item) and has_gold(item)]
    data_by_id = index_by_id(public)
    holdout_ids = make_holdout_ids(public, args)
    raw_by_id = rows_by_id(args.raw_responses)
    candidate_by_id = rows_by_id(args.candidate_responses)
    baseline_by_id = first_by_id(args.baseline)
    rng = random.Random(args.seed)
    judger = Judger(strict_extract=False)

    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    skipped = Counter()
    by_target = Counter()

    for item_id, raw_rows in sorted(raw_by_id.items()):
        item = data_by_id.get(item_id)
        if not item:
            skipped["not_public_mcq"] += len(raw_rows)
            continue
        gold = str(item["answer"]).strip().upper()
        for trace_index, raw_row in enumerate(raw_rows[: args.max_traces_per_id]):
            response = str(raw_row.get("response", ""))
            raw_correct = score_item(judger, item, response)
            split = "holdout" if item_id in holdout_ids else "train"
            examples: list[tuple[dict[str, Any], str, str]] = [(item, gold, "")]
            if split == "train" and args.permutations_per_example > 0:
                for _ in range(args.permutations_per_example):
                    examples.append(permute_item(item, gold, rng))
            for aug_index, (aug_item, aug_target, note) in enumerate(examples):
                row = {
                    "example_id": f"{split}-{item_id}-{trace_index}-{aug_index}",
                    "id": item_id,
                    "split": split,
                    "is_mcq": True,
                    "gold": aug_target,
                    "raw_correct": raw_correct,
                    "raw_answer_key": raw_row.get("answer_key") or extract_answer_key(item, response, strict=True),
                    "messages": [
                        *build_messages(
                            aug_item,
                            raw_row,
                            candidate_by_id.get(item_id, []),
                            baseline_by_id.get(item_id),
                            args.max_trace_chars,
                            permuted_note=note,
                        ),
                        {"role": "assistant", "content": target_text(aug_target, args.boxed_target)},
                    ],
                }
                by_target[aug_target] += 1
                (holdout if split == "holdout" else train).append(row)

    manifest = {
        "public": args.public,
        "raw_responses": args.raw_responses,
        "candidate_responses": args.candidate_responses,
        "baseline": args.baseline,
        "holdout_ids": sorted(holdout_ids),
        "train_examples": len(train),
        "holdout_examples": len(holdout),
        "permutations_per_example": args.permutations_per_example,
        "boxed_target": args.boxed_target,
        "skipped": dict(sorted(skipped.items())),
        "target_counts": dict(sorted(by_target.items())),
    }
    return train, holdout, manifest


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    train_path = Path(args.train_output) if args.train_output else out_dir / "train.jsonl"
    holdout_path = Path(args.holdout_output) if args.holdout_output else out_dir / "holdout.jsonl"
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.json"
    train, holdout, manifest = build_dataset(args)
    write_jsonl(train_path, train)
    write_jsonl(holdout_path, holdout)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "locked_holdout_ids.txt").write_text("\n".join(str(x) for x in manifest["holdout_ids"]) + "\n", encoding="utf-8")
    print(f"Wrote {len(train)} train examples to {train_path}")
    print(f"Wrote {len(holdout)} holdout examples to {holdout_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
