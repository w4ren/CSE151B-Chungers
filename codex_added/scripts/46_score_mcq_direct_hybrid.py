# Added by Codex: summarize MCQ direct-best-of-N hybrid results after a run completes.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from math_comp.data import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print compact MCQ-direct hybrid summary files.")
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    selection = json.loads((run_dir / "selection_summary.json").read_text())
    hybrid_rows = read_jsonl(run_dir / "hybrid_v6_mcq_direct_consensus.jsonl")
    correct = sum(bool(row.get("correct")) for row in hybrid_rows)
    mcq = [row for row in hybrid_rows if row.get("is_mcq")]
    print(
        json.dumps(
            {
                "hybrid_overall": {"correct": correct, "total": len(hybrid_rows)},
                "hybrid_mcq": {"correct": sum(bool(row.get("correct")) for row in mcq), "total": len(mcq)},
                "mcq_oracle": selection.get("oracle"),
                "mcq_policies": selection.get("policies"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
