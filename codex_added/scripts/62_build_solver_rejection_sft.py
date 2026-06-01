# Added by Codex: build solver LoRA SFT data from correct self-generated traces.

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
from math_comp.data import has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.prompts import build_messages
from math_comp.scoring import score_item
from math_comp.variants import get_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rejection-sampling solver SFT messages from correct traces.")
    parser.add_argument("--public", default="data/public.jsonl")
    parser.add_argument("--responses", nargs="+", required=True, help="Candidate raw trace JSONL files.")
    parser.add_argument("--holdout-ids-file", default=None)
    parser.add_argument("--holdout-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1523)
    parser.add_argument("--variant", default="final_box_only")
    parser.add_argument("--max-traces-per-id", type=int, default=3)
    parser.add_argument("--exclude-token-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-response-chars", type=int, default=32000)
    parser.add_argument("--output-dir", default="codex_added/data/solver_rejection_lora")
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
    mcq_ids = [int(item["id"]) for item in items if is_mcq(item)]
    free_ids = [int(item["id"]) for item in items if not is_mcq(item)]
    rng.shuffle(mcq_ids)
    rng.shuffle(free_ids)
    mcq_take = min(len(mcq_ids), args.holdout_count // 3)
    free_take = min(len(free_ids), args.holdout_count - mcq_take)
    return set(mcq_ids[:mcq_take] + free_ids[:free_take])


def response_quality(row: dict[str, Any]) -> tuple[int, int]:
    hit_cap = int(bool(row.get("hit_token_limit")))
    generated = int(row.get("generated_tokens") or 0)
    return hit_cap, generated


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    public = [item for item in read_jsonl(args.public) if has_gold(item)]
    data_by_id = index_by_id(public)
    holdout_ids = make_holdout_ids(public, args)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in args.responses:
        for row in read_jsonl(path):
            grouped[int(row["id"])].append({**row, "source_path": path})

    judger = Judger(strict_extract=False)
    variant = get_variant(args.variant)
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    skipped = Counter()

    for item_id, rows in sorted(grouped.items()):
        item = data_by_id.get(item_id)
        if not item:
            skipped["missing_public_item"] += len(rows)
            continue
        correct_rows: list[dict[str, Any]] = []
        for row in rows:
            response = str(row.get("response", ""))
            if args.exclude_token_limit and row.get("hit_token_limit"):
                skipped["hit_token_limit"] += 1
                continue
            if len(response) > args.max_response_chars:
                skipped["too_long"] += 1
                continue
            if not score_item(judger, item, response):
                skipped["not_correct"] += 1
                continue
            correct_rows.append(row)
        correct_rows.sort(key=response_quality)
        split = "holdout" if item_id in holdout_ids else "train"
        for index, row in enumerate(correct_rows[: args.max_traces_per_id]):
            messages = build_messages(item, variant)
            messages.append({"role": "assistant", "content": str(row.get("response", "")).strip()})
            record = {
                "example_id": f"{split}-{item_id}-{index}",
                "id": item_id,
                "split": split,
                "is_mcq": is_mcq(item),
                "source_path": row.get("source_path"),
                "generated_tokens": row.get("generated_tokens"),
                "hit_token_limit": bool(row.get("hit_token_limit")),
                "messages": messages,
            }
            (holdout if split == "holdout" else train).append(record)

    manifest = {
        "public": args.public,
        "responses": args.responses,
        "holdout_ids": sorted(holdout_ids),
        "train_examples": len(train),
        "holdout_examples": len(holdout),
        "skipped": dict(sorted(skipped.items())),
        "variant": args.variant,
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
