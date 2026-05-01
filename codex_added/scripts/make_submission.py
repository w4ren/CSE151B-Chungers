# Added by Codex: Kaggle CSV conversion CLI; not part of the original starter repository.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from math_comp.submission import write_submission_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert response JSONL into competition submission CSV.")
    parser.add_argument("--data", default="data/private.jsonl", help="Private JSONL path, used for id ordering.")
    parser.add_argument("--predictions", required=True, help="JSONL with id and response fields.")
    parser.add_argument("--output", default="submissions/baseline_submission.csv", help="Submission CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_submission_csv(args.data, args.predictions, args.output)
    print(f"Wrote submission CSV to {args.output}")


if __name__ == "__main__":
    main()
