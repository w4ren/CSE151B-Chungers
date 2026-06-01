from __future__ import annotations

import re
from typing import Any


def is_mcq(item: dict[str, Any]) -> bool:
    return bool(item.get("options"))


def answer_slot_count(item: dict[str, Any]) -> int:
    if is_mcq(item):
        return 1
    answer = item.get("answer")
    if isinstance(answer, list):
        return len(answer)
    return str(item.get("question", "")).count("[ANS]") or 1


def option_letters(item: dict[str, Any]) -> set[str]:
    return {chr(ord("A") + index) for index, _ in enumerate(item.get("options") or [])}


def format_options(options: list[Any]) -> str:
    lines: list[str] = []
    for index, option in enumerate(options):
        letter = chr(ord("A") + index)
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def _boxed_contents(text: str) -> list[str]:
    contents: list[str] = []
    marker = "\\boxed{"
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            break
        cursor = index + len(marker)
        depth = 1
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            contents.append(text[index + len(marker) : cursor - 1].strip())
            start = cursor
        else:
            break
    return contents


def _strip_model_markup(text: str) -> str:
    text = str(text or "").replace("<|im_end|>", " ").replace("<|endoftext|>", " ")
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def extract_answer_key(item: dict[str, Any], text: str) -> str:
    text = _strip_model_markup(text)
    boxed = _boxed_contents(text)
    candidate = boxed[-1] if boxed else text
    if is_mcq(item):
        valid = option_letters(item)
        for pattern in [
            r"^\s*([A-Z])\s*$",
            r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])\b",
            r"\b([A-Z])\b",
        ]:
            for match in re.finditer(pattern, candidate.upper()):
                letter = match.group(1)
                if letter in valid:
                    return letter
        return ""
    return candidate.strip()


def normalize_final_response(item: dict[str, Any], text: str) -> str:
    text = _strip_model_markup(text)
    if is_mcq(item):
        key = extract_answer_key(item, text)
        return f"\\boxed{{{key}}}" if key else "\\boxed{}"

    boxed = _boxed_contents(text)
    if boxed:
        answer = boxed[-1]
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        answer = lines[-1] if lines else text.strip()
        answer = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", answer, flags=re.IGNORECASE)
    answer = " ".join(answer.split())
    return f"\\boxed{{{answer}}}"


def format_contract(item: dict[str, Any]) -> str:
    if is_mcq(item):
        letters = ", ".join(sorted(option_letters(item)))
        return f"Output exactly one option letter from {{{letters}}} inside \\boxed{{}}."
    slots = answer_slot_count(item)
    return f"Output exactly {slots} answer field(s), comma-separated and in [ANS] order, inside \\boxed{{}}."
