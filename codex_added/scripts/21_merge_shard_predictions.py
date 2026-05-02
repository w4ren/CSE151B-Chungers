# Added by Codex: validate and merge selected prediction shards.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from math_comp.data import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge prediction JSONL shards in dataset id order.")
    parser.add_argument("--data", default="data/private.jsonl", help="Full dataset JSONL used for id ordering.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N dataset rows for partial merges.")
    parser.add_argument("--limit", type=int, default=None, help="Limit merged rows after offset.")
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Shard selected.jsonl files to merge.",
    )
    parser.add_argument("--output", default="codex_added/results/best_private_8gpu/selected.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    if args.offset:
        data = data[args.offset :]
    if args.limit is not None:
        data = data[: args.limit]
    expected_ids = [int(item["id"]) for item in data]
    expected_id_set = set(expected_ids)

    by_id: dict[int, dict[str, Any]] = {}
    sources: dict[int, str] = {}
    for prediction_path in args.predictions:
        path = Path(prediction_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing shard prediction file: {path}")
        for row in read_jsonl(path):
            item_id = int(row["id"])
            if item_id not in expected_id_set:
                raise ValueError(f"{path} contains id {item_id}, which is not present in {args.data}")
            if item_id in by_id:
                raise ValueError(
                    f"Duplicate prediction for id {item_id}: {sources[item_id]} and {path}"
                )
            by_id[item_id] = {**row, "id": item_id, "response": str(row.get("response", ""))}
            sources[item_id] = str(path)

    missing = [item_id for item_id in expected_ids if item_id not in by_id]
    if missing:
        preview = ", ".join(str(item_id) for item_id in missing[:10])
        raise ValueError(f"Missing predictions for {len(missing)} ids: {preview}")

    merged = [by_id[item_id] for item_id in expected_ids]
    write_jsonl(args.output, merged)
    print(f"Merged {len(merged)} predictions from {len(args.predictions)} shard file(s) into {args.output}")


if __name__ == "__main__":
    main()
