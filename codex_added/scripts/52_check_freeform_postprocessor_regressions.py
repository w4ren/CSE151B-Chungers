from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from judger import Judger
from math_comp.data import index_by_id, read_jsonl
from math_comp.final_answer import normalize_final_response
from math_comp.scoring import score_item

DATA = CODEX_ROOT / "job_data/public_second100_freeform_stratified50_v6_eval.jsonl"
FIXED_IDS = [
    100,
    102,
    103,
    105,
    113,
    125,
    126,
    127,
    139,
    150,
    151,
    159,
    161,
    164,
    171,
    177,
    191,
    195,
    198,
]


def load_v5_module():
    path = CODEX_ROOT / "scripts/40_deterministic_numeric_finalizer.py"
    spec = importlib.util.spec_from_file_location("v5_numeric_finalizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    v5 = load_v5_module()
    items = index_by_id(read_jsonl(DATA))
    judger = Judger(strict_extract=False)
    failures: list[tuple[int, str]] = []

    for item_id in FIXED_IDS:
        item = items[item_id]
        result = v5.recompute_slots(item)
        if result is None:
            failures.append((item_id, "no recomputer matched"))
            continue
        slots, reason = result
        slots, _ = v5.canonicalize_existing_slots(item, slots)
        response = normalize_final_response(item, v5.box(slots))
        if not score_item(judger, item, response):
            failures.append((item_id, f"{reason} produced {response}"))

    if failures:
        for item_id, message in failures:
            print(f"FAIL {item_id}: {message}")
        raise SystemExit(1)

    print(f"OK: {len(FIXED_IDS)} freeform postprocessor regressions pass")


if __name__ == "__main__":
    main()
