# Added by Codex: response scoring CLI; not part of the original starter repository.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from math_comp.data import read_jsonl, write_jsonl
from math_comp.scoring import score_predictions, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score generated responses against a labeled JSONL file.")
    parser.add_argument("--data", default="data/public.jsonl", help="Labeled public JSONL path.")
    parser.add_argument("--predictions", required=True, help="JSONL with id and response fields.")
    parser.add_argument("--output", default=None, help="Optional scored JSONL output path.")
    parser.add_argument("--strict-extract", action="store_true", help="Disable loose answer extraction fallbacks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    predictions = read_jsonl(args.predictions)
    results = score_predictions(data, predictions, strict_extract=args.strict_extract)
    print(json.dumps(summarize_results(results), indent=2, sort_keys=True))

    if args.output:
        write_jsonl(args.output, results)
        print(f"Wrote scored results to {args.output}")


if __name__ == "__main__":
    main()
