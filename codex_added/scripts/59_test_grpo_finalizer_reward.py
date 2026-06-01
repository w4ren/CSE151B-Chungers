# Added by Codex: tiny reward sanity checks for the GRPO finalizer.

from __future__ import annotations

import sys
from pathlib import Path

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from math_comp.grpo_finalizer import score_finalizer_completion


def assert_gt(left: float, right: float, message: str) -> None:
    if not left > right:
        raise AssertionError(f"{message}: expected {left} > {right}")


def main() -> None:
    mcq = {
        "id": 1,
        "question": "What is 2+2?",
        "options": ["3", "4", "5", "6"],
        "answer": "B",
    }
    good_mcq = score_finalizer_completion(mcq, "\\boxed{B}", raw_extracted_answer="B")
    text_mcq = score_finalizer_completion(mcq, "\\boxed{4}", raw_extracted_answer="B")
    prose_mcq = score_finalizer_completion(mcq, "The answer is B because 2+2=4.", raw_extracted_answer="B")
    assert_gt(good_mcq["reward"], text_mcq["reward"], "letter MCQ should beat option-text/value output")
    assert_gt(good_mcq["reward"], prose_mcq["reward"], "clean MCQ should beat prose")

    multi = {
        "id": 2,
        "question": "Fill two blanks: first [ANS], second [ANS].",
        "answer": ["x+1", "7"],
    }
    good_multi = score_finalizer_completion(multi, "\\boxed{x+1, 7}")
    one_slot = score_finalizer_completion(multi, "\\boxed{x+1}")
    wrong_multi = score_finalizer_completion(multi, "\\boxed{x+1, 8}")
    assert_gt(good_multi["reward"], one_slot["reward"], "complete multi-slot should beat missing slot")
    assert_gt(wrong_multi["reward"], one_slot["reward"], "partial multi-slot credit should beat missing slot")
    print("GRPO finalizer reward sanity checks passed.")


if __name__ == "__main__":
    main()
