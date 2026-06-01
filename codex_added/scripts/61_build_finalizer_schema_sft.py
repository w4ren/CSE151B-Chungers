# Added by Codex: build hard-negative finalizer/schema LoRA SFT data.

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
from math_comp.data import answer_slot_count, has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.grpo_finalizer import build_finalizer_messages, expected_answer_kind
from math_comp.scoring import extract_answer_key, score_item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build finalizer/schema SFT data with hard negatives.")
    parser.add_argument("--public", default="data/public.jsonl")
    parser.add_argument("--raw-responses", nargs="+", required=True)
    parser.add_argument("--current-finalizer", default=None, help="Existing finalizer outputs used as context and hard negatives.")
    parser.add_argument("--holdout-ids-file", default=None)
    parser.add_argument("--holdout-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1522)
    parser.add_argument("--max-trace-chars", type=int, default=6000)
    parser.add_argument("--max-traces-per-id", type=int, default=3)
    parser.add_argument("--synthetic-corruptions-per-row", type=int, default=2)
    parser.add_argument("--oversample-multislot", type=int, default=2)
    parser.add_argument("--include-mcq", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default="codex_added/data/finalizer_schema_lora")
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
    grouped: dict[str, list[int]] = defaultdict(list)
    for item in items:
        if not args.include_mcq and is_mcq(item):
            continue
        grouped[expected_answer_kind(item)].append(int(item["id"]))
    selected: set[int] = set()
    for ids in grouped.values():
        rng.shuffle(ids)
        take = max(1, round(args.holdout_count * len(ids) / max(1, len(items))))
        selected.update(ids[:take])
    if len(selected) < args.holdout_count:
        leftovers = [int(item["id"]) for item in items if int(item["id"]) not in selected]
        rng.shuffle(leftovers)
        selected.update(leftovers[: args.holdout_count - len(selected)])
    return set(sorted(selected)[: args.holdout_count])


def boxed_answer(item: dict[str, Any]) -> str:
    answer = item["answer"]
    if is_mcq(item):
        content = str(answer).strip().upper()
    elif isinstance(answer, list):
        content = ", ".join(str(part).strip() for part in answer)
    else:
        content = str(answer).strip()
    return "\\boxed{" + content + "}"


def corruptions(item: dict[str, Any], target: str, rng: random.Random) -> list[tuple[str, str]]:
    answer = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
    inside = target.removeprefix("\\boxed{").removesuffix("}")
    out: list[tuple[str, str]] = [
        ("missing_box", inside),
        ("extra_prose", f"The final answer is {target}."),
    ]
    if is_mcq(item):
        letter = str(item["answer"]).strip().upper()
        options = item.get("options") or []
        index = ord(letter) - 65
        if 0 <= index < len(options):
            out.append(("mcq_option_text", f"\\boxed{{{options[index]}}}"))
        wrong = chr(65 + ((index + 1) % max(1, len(options)))) if options else "A"
        if wrong != letter:
            out.append(("mcq_wrong_letter", f"\\boxed{{{wrong}}}"))
        return out

    if len(answer) > 1:
        out.append(("missing_last_slot", "\\boxed{" + ", ".join(str(part) for part in answer[:-1]) + "}"))
        shuffled = list(answer)
        rng.shuffle(shuffled)
        if shuffled != answer:
            out.append(("wrong_slot_order", "\\boxed{" + ", ".join(str(part) for part in shuffled) + "}"))
    if re.search(r"\d+\.\d+", inside):
        out.append(("decimal_truncated", "\\boxed{" + re.sub(r"(\d+\.\d{2})\d+", r"\1", inside) + "}"))
    if "/" in inside or "\\frac" in inside:
        out.append(("fraction_decimal_confusion", "\\boxed{" + inside.replace("\\frac", "frac") + "}"))
    lower = str(item.get("question", "")).lower()
    if "yes" in lower or "no" in lower:
        out.append(("yes_no_case", "\\boxed{" + inside.lower() + "}"))
    return out


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


def build_row(
    item: dict[str, Any],
    trace: str,
    target: str,
    split: str,
    source_kind: str,
    raw_extracted: str = "",
    current_answer: str = "",
    max_trace_chars: int = 6000,
) -> dict[str, Any]:
    return {
        "id": int(item["id"]),
        "split": split,
        "is_mcq": is_mcq(item),
        "expected_answer_kind": expected_answer_kind(item),
        "num_ans_slots": answer_slot_count(item),
        "source_kind": source_kind,
        "target_response": target,
        "messages": [
            *build_finalizer_messages(
                item,
                trace,
                raw_extracted_answer=raw_extracted,
                current_finalizer_answer=current_answer,
                max_trace_chars=max_trace_chars,
            ),
            {"role": "assistant", "content": target},
        ],
    }


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    public = [item for item in read_jsonl(args.public) if has_gold(item) and (args.include_mcq or not is_mcq(item))]
    data_by_id = index_by_id(public)
    holdout_ids = make_holdout_ids(public, args)
    raw_by_id = rows_by_id(args.raw_responses)
    current_by_id = first_by_id(args.current_finalizer)
    rng = random.Random(args.seed)
    judger = Judger(strict_extract=False)

    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    source_counts = Counter()

    for item_id, raw_rows in sorted(raw_by_id.items()):
        item = data_by_id.get(item_id)
        if not item:
            continue
        split = "holdout" if item_id in holdout_ids else "train"
        target = boxed_answer(item)
        current_answer = str(current_by_id.get(item_id, {}).get("response", ""))
        for raw in raw_rows[: args.max_traces_per_id]:
            trace = str(raw.get("response", ""))
            raw_extracted = str(raw.get("answer_key") or extract_answer_key(item, trace, strict=True))
            repeat = args.oversample_multislot if answer_slot_count(item) > 1 and split == "train" else 1
            for _ in range(repeat):
                row = build_row(item, trace, target, split, "raw_trace", raw_extracted, current_answer, args.max_trace_chars)
                (holdout if split == "holdout" else train).append(row)
                source_counts[row["source_kind"]] += 1

            if split == "train":
                current_wrong = False
                if current_answer:
                    try:
                        current_wrong = not score_item(judger, item, current_answer)
                    except Exception:
                        current_wrong = True
                if current_wrong:
                    row = build_row(item, current_answer, target, split, "current_finalizer_hard_negative", raw_extracted, current_answer, args.max_trace_chars)
                    train.append(row)
                    source_counts[row["source_kind"]] += 1
                for category, bad in corruptions(item, target, rng)[: args.synthetic_corruptions_per_row]:
                    row = build_row(item, f"Reasoning omitted. Final answer attempt: {bad}", target, split, f"synthetic_{category}", raw_extracted, bad, args.max_trace_chars)
                    train.append(row)
                    source_counts[row["source_kind"]] += 1

    manifest = {
        "public": args.public,
        "raw_responses": args.raw_responses,
        "current_finalizer": args.current_finalizer,
        "holdout_ids": sorted(holdout_ids),
        "train_examples": len(train),
        "holdout_examples": len(holdout),
        "source_counts": dict(sorted(source_counts.items())),
        "include_mcq": args.include_mcq,
        "synthetic_corruptions_per_row": args.synthetic_corruptions_per_row,
        "oversample_multislot": args.oversample_multislot,
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


if __name__ == "__main__":
    main()
