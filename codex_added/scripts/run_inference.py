# Added by Codex: baseline generation CLI; not part of the original starter repository.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from math_comp.data import has_gold, load_yaml_config, read_jsonl, write_jsonl
from math_comp.inference import generate_responses
from math_comp.scoring import score_predictions, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses with Qwen3-4B-Thinking-2507.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="YAML config path.")
    parser.add_argument("--data", default=None, help="Override input JSONL path.")
    parser.add_argument("--output", default=None, help="Override output JSONL path.")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default=None, help="Override backend.")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for a smoke test.")
    parser.add_argument("--no-score", action="store_true", help="Do not score even when answers are present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.backend:
        config.setdefault("model", {})["backend"] = args.backend

    paths = config.get("paths", {})
    data_path = args.data or paths.get("public_data", "data/public.jsonl")
    output_path = args.output or paths.get("results", "results/baseline_results.jsonl")

    items = read_jsonl(data_path)
    if args.limit is not None:
        items = items[: args.limit]

    records = generate_responses(items, config)
    should_score = not args.no_score and all(has_gold(item) for item in items)
    if should_score:
        records = score_predictions(items, records)
        print(json.dumps(summarize_results(records), indent=2, sort_keys=True))
    else:
        print(f"Generated {len(records)} responses without scoring.")

    write_jsonl(output_path, records)
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
