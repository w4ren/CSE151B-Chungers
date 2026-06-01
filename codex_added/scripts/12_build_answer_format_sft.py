# Added by Codex: answer-format SFT data builder; not part of the original starter repository.

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

from math_comp.data import is_mcq, read_jsonl
from math_comp.prompts import build_messages
from math_comp.variants import get_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build public-set SFT data for final-answer format discipline.")
    parser.add_argument("--public", default="data/public.jsonl", help="Public JSONL with answers.")
    parser.add_argument("--output", default="codex_added/data/sft_answer_format.jsonl", help="Output JSONL.")
    parser.add_argument("--variant", default="final_box_only", help="Prompt variant to use for messages.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    return parser.parse_args()


def boxed_answer(item: dict[str, Any]) -> str:
    answer = item["answer"]
    if is_mcq(item):
        content = str(answer).strip().upper()
    elif isinstance(answer, list):
        content = ", ".join(str(part).strip() for part in answer)
    else:
        content = str(answer).strip()
    return f"\\boxed{{{content}}}"


def main() -> None:
    args = parse_args()
    items = read_jsonl(args.public)
    if args.limit is not None:
        items = items[: args.limit]

    prompt_config = get_variant(args.variant)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for item in items:
            messages = build_messages(item, prompt_config)
            messages.append(
                {
                    "role": "assistant",
                    "content": f"\n</think>\n\n{boxed_answer(item)}",
                }
            )
            record = {
                "id": int(item["id"]),
                "is_mcq": is_mcq(item),
                "messages": messages,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} SFT examples to {out_path}")


if __name__ == "__main__":
    main()
