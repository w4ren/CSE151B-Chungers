# Added by Codex: build an MCQ-only slice from a mixed competition JSONL.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from math_comp.data import is_mcq, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MCQ rows from a mixed JSONL dataset.")
    parser.add_argument("--data", required=True, help="Input mixed JSONL.")
    parser.add_argument("--output", required=True, help="Output MCQ-only JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit after MCQ filtering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [row for row in read_jsonl(args.data) if is_mcq(row)]
    if args.limit is not None:
        rows = rows[: args.limit]
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} MCQ rows to {args.output}")


if __name__ == "__main__":
    main()
