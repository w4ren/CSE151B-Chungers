# Added by Codex: build direct public-MCQ SFT data without trace/candidate grading context.

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from math_comp.data import has_gold, is_mcq, read_jsonl, write_jsonl
from math_comp.prompts import format_options


SYSTEM = (
    "You are an expert mathematician answering a multiple-choice math problem. "
    "Read the problem and options, then output exactly one option letter inside \\boxed{}. "
    "Do not output option text. Do not explain."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build direct public-MCQ-only chat SFT data.")
    parser.add_argument("--public", default="data/public.jsonl")
    parser.add_argument("--output-dir", default="codex_added/data/public_mcq_direct_lora")
    parser.add_argument("--train-output", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--seed", type=int, default=1561)
    parser.add_argument(
        "--permutations-per-example",
        type=int,
        default=0,
        help="Optional option-order augmentation using public gold only.",
    )
    return parser.parse_args()


def user_prompt(item: dict[str, Any]) -> str:
    return (
        "Problem:\n"
        f"{item['question']}\n\n"
        "Options:\n"
        f"{format_options(item.get('options') or [])}\n\n"
        "Final answer only:"
    )


def boxed(letter: str) -> str:
    return "\\boxed{" + letter.strip().upper() + "}"


def permute_item(item: dict[str, Any], target: str, rng: random.Random) -> tuple[dict[str, Any], str]:
    options = list(item.get("options") or [])
    if not options:
        return item, target
    gold_index = ord(target.strip().upper()) - 65
    if gold_index < 0 or gold_index >= len(options):
        return item, target
    perm = list(range(len(options)))
    rng.shuffle(perm)
    if perm == list(range(len(options))):
        perm = perm[1:] + perm[:1]
    new_options = [options[index] for index in perm]
    new_target = chr(65 + perm.index(gold_index))
    return {**item, "options": new_options}, new_target


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    target_counts: Counter[str] = Counter()
    source_items = [item for item in read_jsonl(args.public) if is_mcq(item) and has_gold(item)]

    for item in source_items:
        item_id = int(item["id"])
        gold = str(item["answer"]).strip().upper()
        examples = [(item, gold)]
        for _ in range(max(0, args.permutations_per_example)):
            examples.append(permute_item(item, gold, rng))
        for aug_index, (example_item, target) in enumerate(examples):
            target_counts[target] += 1
            rows.append(
                {
                    "example_id": f"public-mcq-direct-{item_id}-{aug_index}",
                    "id": item_id,
                    "is_mcq": True,
                    "gold": target,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user_prompt(example_item)},
                        {"role": "assistant", "content": boxed(target)},
                    ],
                }
            )

    rng.shuffle(rows)
    manifest = {
        "public": args.public,
        "selection": "public MCQ rows with gold labels only; no raw traces, candidates, option grading, or private labels",
        "source_mcq_items": len(source_items),
        "train_examples": len(rows),
        "permutations_per_example": args.permutations_per_example,
        "target_counts": dict(sorted(target_counts.items())),
    }
    return rows, manifest


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    train_path = Path(args.train_output) if args.train_output else out_dir / "train.jsonl"
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.json"
    rows, manifest = build_dataset(args)
    write_jsonl(train_path, rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} train examples to {train_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
