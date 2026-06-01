# Added by Codex: shared helpers for GRPO finalizer data, prompts, and rewards.

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from judger import Judger
from math_comp.data import answer_slot_count, has_gold, is_mcq
from math_comp.final_answer import final_answer_diagnostics, normalize_final_response
from math_comp.prompts import format_options
from math_comp.scoring import FREE_FORM_EXCLUDES, extract_answer_key, score_item


@dataclass(frozen=True)
class FinalizerRewardWeights:
    correct: float = 1.0
    raw_completion_correct: float = 0.20
    mcq_letter_format: float = 0.05
    slot_count: float = 0.05
    partial_slots: float = 0.30
    clean_box: float = 0.01
    missing_slots: float = -0.05
    extra_slots: float = -0.05
    mcq_option_text: float = -0.08
    long_prose: float = -0.10
    raw_mcq_flip: float = -0.10


def boxed(content: str) -> str:
    return "\\boxed{" + str(content).strip() + "}"


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(completion)


def compact_trace(trace: str, max_chars: int, head_chars: int = 1000) -> str:
    trace = str(trace or "").strip()
    if max_chars <= 0 or len(trace) <= max_chars:
        return trace
    head_chars = max(0, min(head_chars, max_chars // 2))
    tail_chars = max_chars - head_chars
    omitted = len(trace) - head_chars - tail_chars
    return (
        trace[:head_chars].rstrip()
        + f"\n\n[... omitted {omitted} trace characters ...]\n\n"
        + trace[-tail_chars:].lstrip()
    )


def expected_answer_kind(item: dict[str, Any]) -> str:
    if is_mcq(item):
        return "mcq_letter"
    question = str(item.get("question", "")).lower()
    slots = answer_slot_count(item)
    if slots > 1:
        return "multi_slot"
    if re.search(r"\btrue\s+or\s+false\b|\btrue/false\b", question):
        return "true_false"
    if re.search(r"\byes\s+or\s+no\b|\byes/no\b", question):
        return "yes_no"
    if re.search(r"\binteger\b|whole number|nearest integer", question):
        return "integer"
    if re.search(r"\bdecimal\b|nearest tenth|nearest hundredth|decimal places?", question):
        return "decimal"
    if re.search(r"\bfraction\b|lowest terms|simplest form|\\frac", question):
        return "exact_fraction"
    if re.search(r"\blist\b|find all|all solutions|ordered", question):
        return "ordered_list"
    if re.search(r"\bexpression\b|equation|in terms of|exact form|\\sqrt|\\pi", question):
        return "expression"
    return "unknown"


def format_profile(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "is_mcq": is_mcq(item),
        "num_ans_slots": answer_slot_count(item),
        "expected_answer_kind": expected_answer_kind(item),
    }


def finalizer_system_prompt(item: dict[str, Any]) -> str:
    if is_mcq(item):
        return (
            "You are an MCQ answer finalizer. Output exactly one uppercase option letter. "
            "Do not output option text. If the solver trace clearly supports a letter, "
            "preserve it unless it contradicts the problem."
        )
    return (
        "You are a free-form answer finalizer. Output only the requested answer values. "
        "If there are multiple [ANS] blanks, output exactly one value per blank in the same order. "
        "Match requested rounding, exact form, sign, units, and notation. Do not include explanation."
    )


def problem_text(item: dict[str, Any]) -> str:
    text = str(item.get("question", ""))
    if item.get("options"):
        text += "\n\nOptions:\n" + format_options(item["options"])
    return text


def build_finalizer_messages(
    item: dict[str, Any],
    solver_trace: str,
    raw_extracted_answer: str = "",
    current_finalizer_answer: str = "",
    max_trace_chars: int = 6000,
) -> list[dict[str, str]]:
    profile = format_profile(item)
    trace = compact_trace(solver_trace, max_trace_chars)
    if is_mcq(item):
        task = (
            "Final answer letter:\n"
            "- Output exactly one uppercase option letter.\n"
            "- Do not output option text.\n"
            "- Do not explain."
        )
    else:
        task = (
            "Final answer:\n"
            f"- Required number of answer slots: {profile['num_ans_slots']}.\n"
            "- Output exactly one value per slot in the original order.\n"
            "- Do not explain."
        )

    user = (
        "Problem:\n"
        f"{problem_text(item)}\n\n"
        "Solver trace:\n"
        f"{trace}\n\n"
        "Raw extracted answer:\n"
        f"{raw_extracted_answer or '<none>'}\n\n"
        "Current finalizer answer:\n"
        f"{current_finalizer_answer or '<none>'}\n\n"
        "Formatting requirements:\n"
        f"{json.dumps(profile, sort_keys=True)}\n\n"
        f"{task}"
    )
    return [
        {"role": "system", "content": finalizer_system_prompt(item)},
        {"role": "user", "content": user},
    ]


def direct_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    forced = "<|im_start|>assistant\n<think>\n"
    direct = "<|im_start|>assistant\n"
    if text.endswith(forced):
        text = text[: -len(forced)] + direct
    return text


def exact_letter_output(text: str, option_count: int) -> str:
    allowed = "".join(chr(65 + index) for index in range(option_count))
    stripped = str(text).strip()
    match = re.fullmatch(rf"([A-{allowed[-1]}])", stripped) if allowed else None
    if match:
        return match.group(1)
    match = re.fullmatch(rf"\\boxed\{{([A-{allowed[-1]}])\}}", stripped) if allowed else None
    return match.group(1) if match else ""


def clean_boxed_output(text: str) -> bool:
    stripped = str(text).strip()
    if not re.fullmatch(r"\\boxed\{.+\}", stripped, flags=re.DOTALL):
        return False
    return stripped.count("\\boxed") == 1 and "\n" not in stripped


def contains_option_text(item: dict[str, Any], completion: str) -> bool:
    if not is_mcq(item):
        return False
    lowered = str(completion).lower()
    for option in item.get("options") or []:
        text = str(option).strip().lower()
        if len(text) >= 3 and text in lowered:
            return True
    return False


def contains_long_prose(completion: str) -> bool:
    text = str(completion).strip()
    if len(text) > 180:
        return True
    if "</think>" in text or "<think>" in text:
        return True
    if "\n" in text and not clean_boxed_output(text):
        return True
    prose_markers = ["because", "therefore", "we need", "the answer is", "so the answer"]
    lowered = text.lower()
    return any(marker in lowered for marker in prose_markers) and not re.fullmatch(r"\\boxed\{[^{}]+\}", text)


def predicted_slots(item: dict[str, Any], response: str, judger: Judger) -> list[str]:
    if is_mcq(item):
        key = extract_answer_key(item, response, strict=True)
        return [key] if key else []
    extracted = judger.extract_ans(response)
    return judger.split_by_comma(extracted) if extracted else []


def partial_slot_score(item: dict[str, Any], response: str, judger: Judger) -> tuple[int, int, float]:
    if is_mcq(item) or not has_gold(item):
        return 0, 0, 0.0
    gold = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
    pred = predicted_slots(item, response, judger)
    total = len(gold)
    if not pred or total == 0:
        return 0, total, 0.0
    correct = 0
    for pred_slot, gold_slot in zip(pred, gold):
        try:
            pred_norm = judger.norm_ans_str(str(pred_slot))
            gold_norm = judger.norm_ans_str(str(gold_slot))
            if judger.is_equal(pred_norm, gold_norm, options=[], exclude=FREE_FORM_EXCLUDES):
                correct += 1
        except Exception:
            pass
    return correct, total, correct / total


def safe_score(item: dict[str, Any], response: str, judger: Judger) -> bool:
    if not has_gold(item):
        return False
    try:
        return score_item(judger, item, response)
    except Exception:
        return False


def score_finalizer_completion(
    item: dict[str, Any],
    completion: Any,
    raw_extracted_answer: str = "",
    weights: FinalizerRewardWeights | None = None,
) -> dict[str, Any]:
    weights = weights or FinalizerRewardWeights()
    weight_dict = asdict(weights)
    text = completion_text(completion).strip()
    normalized = normalize_final_response(item, text)
    judger = Judger(strict_extract=False)
    strict_judger = Judger(strict_extract=True)

    correct = safe_score(item, normalized, judger)
    raw_completion_correct = safe_score(item, text, judger)
    diagnostics = final_answer_diagnostics(item, normalized)
    expected_slots = int(diagnostics.get("expected_slots") or answer_slot_count(item))
    actual_slots = int(diagnostics.get("parsed_slots") or 0)
    answer_key = extract_answer_key(item, normalized, strict=True)

    components: dict[str, float] = {}
    if correct:
        components["correct"] = weights.correct
    if raw_completion_correct:
        components["raw_completion_correct"] = weights.raw_completion_correct
    if is_mcq(item) and exact_letter_output(text, len(item.get("options") or [])):
        components["mcq_letter_format"] = weights.mcq_letter_format
    if not is_mcq(item) and expected_slots and actual_slots == expected_slots:
        components["slot_count"] = weights.slot_count
    if clean_boxed_output(text):
        components["clean_box"] = weights.clean_box

    slot_correct, slot_total, slot_fraction = partial_slot_score(item, normalized, strict_judger)
    if slot_total > 1 and slot_fraction:
        components["partial_slots"] = weights.partial_slots * slot_fraction

    if not is_mcq(item) and expected_slots and actual_slots < expected_slots:
        components["missing_slots"] = weights.missing_slots
    if not is_mcq(item) and expected_slots and actual_slots > expected_slots:
        components["extra_slots"] = weights.extra_slots
    if is_mcq(item) and contains_option_text(item, text):
        components["mcq_option_text"] = weights.mcq_option_text
    if contains_long_prose(text):
        components["long_prose"] = weights.long_prose

    raw_key = str(raw_extracted_answer or "").strip().upper()
    raw_was_correct = False
    if is_mcq(item) and raw_key:
        raw_was_correct = safe_score(item, boxed(raw_key), judger)
        if raw_was_correct and (not correct or answer_key.upper() != raw_key):
            components["raw_mcq_flip"] = weights.raw_mcq_flip

    reward = float(sum(components.values()))
    return {
        "reward": reward,
        "reward_components": components,
        "reward_weights": weight_dict,
        "completion": text,
        "normalized_answer": normalized,
        "answer_key": answer_key,
        "gold": item.get("answer"),
        "correct": correct,
        "raw_completion_correct": raw_completion_correct,
        "raw_extracted_answer": raw_key,
        "raw_extracted_was_correct": raw_was_correct,
        "expected_slots": expected_slots,
        "parsed_slots": actual_slots,
        "slot_correct": slot_correct,
        "slot_total": slot_total,
    }
