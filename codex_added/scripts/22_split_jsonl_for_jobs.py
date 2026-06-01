# Added by Codex: split a JSONL dataset into balanced job input files.

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

from math_comp.data import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a competition JSONL into contiguous job chunks.")
    parser.add_argument("--data", default="data/private.jsonl", help="Input JSONL to split.")
    parser.add_argument("--out-dir", default="codex_added/job_data", help="Directory for split JSONL files.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix; defaults to input stem.")
    parser.add_argument("--num-jobs", type=int, default=2, help="Number of balanced chunks to create.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest JSON path; defaults to OUT_DIR/PREFIX_manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_jobs <= 0:
        raise ValueError("--num-jobs must be positive")

    data_path = Path(args.data)
    rows = read_jsonl(data_path)
    if not rows:
        raise ValueError(f"No rows found in {data_path}")

    prefix = args.prefix or data_path.stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(rows)
    base_size = total // args.num_jobs
    remainder = total % args.num_jobs
    offset = 0
    chunks: list[dict[str, Any]] = []

    for job_index in range(args.num_jobs):
        count = base_size + (1 if job_index < remainder else 0)
        chunk_rows = rows[offset : offset + count]
        chunk_path = out_dir / f"{prefix}_job{job_index}.jsonl"
        write_jsonl(chunk_path, chunk_rows)
        chunks.append(
            {
                "job_index": job_index,
                "path": str(chunk_path),
                "offset": offset,
                "rows": count,
                "first_id": int(chunk_rows[0]["id"]) if chunk_rows else None,
                "last_id": int(chunk_rows[-1]["id"]) if chunk_rows else None,
            }
        )
        offset += count

    manifest_path = Path(args.manifest) if args.manifest else out_dir / f"{prefix}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(data_path),
        "total_rows": total,
        "num_jobs": args.num_jobs,
        "chunks": chunks,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.num_jobs} split file(s) for {total} rows to {out_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
