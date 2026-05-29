# Added by Codex: final-answer formatting helpers; not part of the original starter repository.

from __future__ import annotations

import re
from typing import Any

from judger import Judger
from math_comp.data import answer_slot_count, is_mcq
from math_comp.scoring import extract_answer_key, extract_mcq_letter


BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")


def final_format_contract(item: dict[str, Any]) -> str:
    if is_mcq(item):
        option_count = len(item.get("options") or [])
        letters = ", ".join(chr(65 + index) for index in range(option_count))
        return (
            "Output exactly one line: \\boxed{X}, where X is exactly one option "
            f"letter from {{{letters}}}. Choose the letter whose option text matches "
            "the required form most directly; do not choose an alternate equivalent "
            "form if another option uses the requested notation. Do not include option "
            "text or prose."
        )

    slots = answer_slot_count(item)
    plural = "answer" if slots == 1 else "ordered answers"
    return (
        f"Output exactly one line: \\boxed{{...}} containing exactly {slots} {plural}. "
        "There must be one comma-separated entry for every [ANS] blank in the problem, "
        "in the original slot order. If the trace filled a table, include every table "
        "blank, not just the last statistic. Preserve the most precise value present "
        "in the trace unless the problem explicitly asks for rounding to a specified "
        "place. Do not add prose before or after the box."
    )


def last_boxed_content(response: str) -> str:
    matches = BOXED_PATTERN.findall(response)
    return matches[-1].strip() if matches else ""


def normalize_final_response(item: dict[str, Any], response: str) -> str:
    response = str(response).strip()
    if is_mcq(item):
        letter = extract_mcq_letter(response, len(item.get("options") or []))
        return f"\\boxed{{{letter}}}" if letter else response

    boxed = last_boxed_content(response)
    if boxed:
        return f"\\boxed{{{boxed}}}"

    key = extract_answer_key(item, response, strict=False)
    if key:
        return f"\\boxed{{{key}}}"
    return response


def final_answer_diagnostics(item: dict[str, Any], response: str) -> dict[str, Any]:
    normalized = normalize_final_response(item, response)
    expected_slots = answer_slot_count(item)
    strict_key = extract_answer_key(item, normalized, strict=True)

    if is_mcq(item):
        parsed_slots = 1 if strict_key else 0
    else:
        extracted = Judger(strict_extract=True).extract_ans(normalized)
        parsed_slots = len(Judger(strict_extract=True).split_by_comma(extracted)) if extracted else 0

    return {
        "expected_slots": expected_slots,
        "parsed_slots": parsed_slots,
        "format_ok": bool(strict_key) and parsed_slots == expected_slots,
    }
