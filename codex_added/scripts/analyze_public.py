# Added by Codex: dataset inspection CLI; not part of the original starter repository.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from math_comp.data import read_jsonl, summarize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a competition JSONL dataset.")
    parser.add_argument("--data", default="data/public.jsonl", help="Path to public/private JSONL data.")
    parser.add_argument("--examples", type=int, default=2, help="Number of example rows to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = read_jsonl(args.data)
    print(json.dumps(summarize_dataset(items), indent=2, sort_keys=True))

    for item in items[: max(args.examples, 0)]:
        preview = dict(item)
        question = str(preview.get("question", ""))
        preview["question"] = question[:300] + ("..." if len(question) > 300 else "")
        print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
