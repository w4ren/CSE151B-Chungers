# Added by Codex: scoring helpers around the starter judger; not part of the original starter repository.

from __future__ import annotations

import re
from typing import Any, Iterable

from judger import Judger
from math_comp.data import has_gold, index_by_id, is_mcq

FREE_FORM_EXCLUDES = ["UOL", "OL", "MCM", "MCS"]


def extract_mcq_letter(response: str, option_count: int | None = None) -> str:
    allowed = {chr(65 + idx) for idx in range(option_count or 26)}
    boxed_matches = re.findall(r"\\boxed\{([^{}]*)\}", response)
    candidates: list[str] = []

    for boxed in boxed_matches:
        candidates.extend(re.findall(r"\b([A-Z])\b", boxed.upper()))

    if not candidates:
        candidates = re.findall(r"\b([A-Z])\b", response.upper())

    for candidate in reversed(candidates):
        if candidate in allowed:
            return candidate
    return ""


def score_mcq(response: str, gold_letter: str, option_count: int | None = None) -> bool:
    return extract_mcq_letter(response, option_count) == gold_letter.strip().upper()


def _single_answer_slot(item: dict[str, Any]) -> bool:
    return str(item.get("question", "")).count("[ANS]") <= 1


def _extract_loose_phrase_answer(item: dict[str, Any], response: str) -> str:
    if not _single_answer_slot(item):
        return ""

    segments = []
    if "</think>" in response:
        before, after = response.rsplit("</think>", 1)
        segments.extend([after, before])
    segments.append(response)

    answer_pattern = re.compile(
        r"(?:final\s+answer|answer|result|sum)\s*(?:is|=|:)\s*"
        r"(?P<answer>\\frac\{[^{}]+\}\{[^{}]+\}|[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?)",
        re.IGNORECASE,
    )
    for segment in segments:
        matches = list(answer_pattern.finditer(segment))
        if not matches:
            continue
        answer = matches[-1].group("answer").strip().strip("$").replace(",", "")
        if answer and "[ANS]" not in answer:
            return answer
    return ""


def extract_answer_key(
    item: dict[str, Any],
    response: str,
    judger: Judger | None = None,
    strict: bool = True,
) -> str:
    """Return a normalized answer key suitable for self-consistency voting."""
    if is_mcq(item):
        return extract_mcq_letter(response, len(item.get("options") or []))

    judger = judger or Judger(strict_extract=strict)
    if not strict:
        phrase_answer = _extract_loose_phrase_answer(item, response)
        if phrase_answer:
            return judger.norm_ans_str(phrase_answer)

    extracted = judger.extract_ans(response)
    if not extracted:
        return ""
    if not strict and ("</think>" in extracted or "[ANS]" in extracted or len(extracted) > 100):
        return ""

    parts = judger.split_by_comma(extracted)
    if not parts:
        return ""
    try:
        normalized = [judger.norm_ans_str(part) for part in parts]
    except Exception:
        normalized = [part.strip() for part in parts]
    return ", ".join(normalized)


def score_item(judger: Judger, item: dict[str, Any], response: str) -> bool:
    if not has_gold(item):
        raise ValueError(f"Item {item.get('id')} does not include a ground-truth answer")

    if is_mcq(item):
        return score_mcq(response, str(item["answer"]), len(item.get("options") or []))

    gold = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
    try:
        extracted = judger.extract_ans(response)
        if not extracted:
            return False
        pred_parts = judger.split_by_comma(extracted)
        if len(pred_parts) != len(gold):
            return False
        pred_norm = [judger.norm_ans_str(part) for part in pred_parts]
        gold_norm = [judger.norm_ans_str(part) for part in gold]
        return all(
            judger.is_equal(pred, expected, options=[], exclude=FREE_FORM_EXCLUDES)
            for pred, expected in zip(pred_norm, gold_norm)
        )
    except Exception:
        return False


def score_predictions(
    data: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    strict_extract: bool = False,
) -> list[dict[str, Any]]:
    data_by_id = index_by_id(data)
    pred_by_id = index_by_id(predictions)
    judger = Judger(strict_extract=strict_extract)
    results: list[dict[str, Any]] = []

    for item_id, item in data_by_id.items():
        pred = pred_by_id.get(item_id, {})
        response = str(pred.get("response", ""))
        correct = score_item(judger, item, response) if has_gold(item) else None
        record = {
            "id": item_id,
            "is_mcq": is_mcq(item),
            "response": response,
        }
        if has_gold(item):
            record["gold"] = item["answer"]
            record["correct"] = correct
        results.append(record)

    return results


def summarize_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in results if row.get("correct") is not None]
    mcq = [row for row in rows if row["is_mcq"]]
    free = [row for row in rows if not row["is_mcq"]]

    def summarize_subset(subset: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(bool(row["correct"]) for row in subset)
        total = len(subset)
        return {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }

    return {
        "overall": summarize_subset(rows),
        "mcq": summarize_subset(mcq),
        "free_form": summarize_subset(free),
    }
