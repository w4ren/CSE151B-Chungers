# Added by Codex: build reproducible audit datasets and raw shards.

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

from math_comp.data import index_by_id, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare public audit datasets and raw prediction shards.")
    parser.add_argument("--public-data", default="data/public.jsonl")
    parser.add_argument(
        "--diagnostic-data",
        default="codex_added/job_data/public_diagnostic_100_offset200_40mcq_40multislot_20hard.jsonl",
    )
    parser.add_argument("--public-first50-raw", required=True)
    parser.add_argument("--public-missing-raw", nargs="+", required=True)
    parser.add_argument("--diagnostic-raw", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--combined-data",
        default="codex_added/job_data/public200_first100_plus_diagnostic100.jsonl",
    )
    parser.add_argument("--extra100-data", default="codex_added/job_data/public_extra100_offset100.jsonl")
    return parser.parse_args()


def read_many(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def require_unique_ids(rows: list[dict[str, Any]], label: str) -> None:
    ids = [int(row["id"]) for row in rows]
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        raise ValueError(f"{label} has duplicate ids: {duplicates[:10]}")


def select_rows_by_ids(rows: list[dict[str, Any]], ids: list[int], label: str) -> list[dict[str, Any]]:
    by_id = index_by_id(rows)
    missing = [item_id for item_id in ids if item_id not in by_id]
    if missing:
        raise ValueError(f"{label} is missing {len(missing)} ids: {missing[:10]}")
    return [by_id[item_id] for item_id in ids]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_for_finalizer"
    raw_dir.mkdir(parents=True, exist_ok=True)

    public_rows = read_jsonl(args.public_data)
    diagnostic_rows = read_jsonl(args.diagnostic_data)
    first100 = public_rows[:100]
    if len(first100) != 100:
        raise ValueError(f"Need 100 public rows, got {len(first100)}")
    combined_data = first100 + diagnostic_rows
    require_unique_ids(combined_data, "combined data")
    write_jsonl(args.combined_data, combined_data)

    used_ids = {int(row["id"]) for row in combined_data}
    extra100 = [row for row in public_rows if int(row["id"]) not in used_ids][:100]
    if len(extra100) != 100:
        raise ValueError(f"Need 100 non-overlapping extra rows, got {len(extra100)}")
    write_jsonl(args.extra100_data, extra100)

    public_raw = read_many([args.public_first50_raw, *args.public_missing_raw])
    diagnostic_raw = read_jsonl(args.diagnostic_raw)
    require_unique_ids(public_raw, "public raw")
    require_unique_ids(diagnostic_raw, "diagnostic raw")
    raw_by_id = index_by_id(public_raw + diagnostic_raw)

    combined_ids = [int(row["id"]) for row in combined_data]
    missing_raw = [item_id for item_id in combined_ids if item_id not in raw_by_id]
    if missing_raw:
        raise ValueError(f"Combined raw is missing {len(missing_raw)} ids: {missing_raw[:20]}")

    combined_raw = [{**raw_by_id[item_id], "id": item_id} for item_id in combined_ids]
    write_jsonl(out_dir / "raw_8k_merged.jsonl", combined_raw)

    # Keep two balanced finalizer shards in combined data order.
    midpoint = len(combined_raw) // 2
    write_jsonl(raw_dir / "shard_00.jsonl", combined_raw[:midpoint])
    write_jsonl(raw_dir / "shard_01.jsonl", combined_raw[midpoint:])

    manifest = {
        "combined_data": args.combined_data,
        "combined_rows": len(combined_data),
        "combined_ids": combined_ids,
        "extra100_data": args.extra100_data,
        "extra100_rows": len(extra100),
        "extra100_ids": [int(row["id"]) for row in extra100],
        "raw_merged": str(out_dir / "raw_8k_merged.jsonl"),
        "raw_shards": [str(raw_dir / "shard_00.jsonl"), str(raw_dir / "shard_01.jsonl")],
        "sources": {
            "public_data": args.public_data,
            "diagnostic_data": args.diagnostic_data,
            "public_first50_raw": args.public_first50_raw,
            "public_missing_raw": args.public_missing_raw,
            "diagnostic_raw": args.diagnostic_raw,
        },
    }
    (out_dir / "audit_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in ("combined_rows", "extra100_rows", "raw_shards")}, indent=2))


if __name__ == "__main__":
    main()
