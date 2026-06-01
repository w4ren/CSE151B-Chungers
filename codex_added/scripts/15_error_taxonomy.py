# Added by Codex: error taxonomy for scored runs; not part of the original starter repository.

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from judger import Judger
from math_comp.data import index_by_id, is_mcq, read_jsonl
from math_comp.scoring import extract_answer_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket scored errors from a public evaluation run.")
    parser.add_argument("--data", default="data/public.jsonl", help="Public JSONL with answers.")
    parser.add_argument("--results", required=True, help="Scored JSONL with id, response, and correct fields.")
    parser.add_argument("--examples-per-bucket", type=int, default=3, help="Stored example ids per bucket.")
    parser.add_argument("--output", default=None, help="Optional JSON summary output.")
    return parser.parse_args()


def answer_count(item: dict[str, Any]) -> int:
    answer = item.get("answer")
    if isinstance(answer, list):
        return len(answer)
    return 1


def predicted_count(judger: Judger, item: dict[str, Any], response: str) -> int:
    if is_mcq(item):
        return 1 if extract_answer_key(item, response, judger) else 0
    extracted = judger.extract_ans(response)
    return len(judger.split_by_comma(extracted)) if extracted else 0


def classify(judger: Judger, item: dict[str, Any], row: dict[str, Any]) -> str:
    if bool(row.get("correct")):
        return "correct"

    response = str(row.get("response", ""))
    key = str(row.get("answer_key") or extract_answer_key(item, response, Judger(strict_extract=True)))
    if not key:
        return "no_extractable_answer"
    if is_mcq(item):
        return "wrong_mcq_letter"
    if predicted_count(judger, item, response) != answer_count(item):
        return "answer_count_mismatch"
    if "\\boxed" not in response:
        return "unboxed_or_loose_answer"
    return "wrong_free_form_value"


def main() -> None:
    args = parse_args()
    data_by_id = index_by_id(read_jsonl(args.data))
    rows = read_jsonl(args.results)
    judger = Judger(strict_extract=False)

    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        item_id = int(row["id"])
        item = data_by_id[item_id]
        bucket = classify(judger, item, row)
        counts[bucket] += 1
        if bucket != "correct" and len(examples[bucket]) < args.examples_per_bucket:
            examples[bucket].append(
                {
                    "id": item_id,
                    "gold": item.get("answer"),
                    "answer_key": row.get("answer_key") or extract_answer_key(item, str(row.get("response", "")), judger),
                    "response_preview": str(row.get("response", ""))[:500],
                }
            )

    summary = {
        "counts": dict(counts.most_common()),
        "examples": examples,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
