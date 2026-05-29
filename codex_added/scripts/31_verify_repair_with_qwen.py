# Added by Codex: third-stage Qwen verifier/repair pass for finalized answers.

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from judger import Judger
from math_comp.data import (
    append_jsonl_record,
    batched,
    has_gold,
    index_by_id,
    is_mcq,
    read_jsonl,
    read_jsonl_complete_prefix,
    write_jsonl,
)
from math_comp.final_answer import normalize_final_response
from math_comp.prompts import format_options
from math_comp.scoring import extract_answer_key, score_item, summarize_results
from math_comp.vllm_helpers import add_vllm_cli_args, make_llm, make_sampling_params, vllm_generated_tokens, vllm_text


REPAIR_SYSTEM = (
    "You are Qwen repairing a math competition final answer for an exact-match evaluator. "
    "Do not solve from scratch. Use only the problem, the previous Qwen reasoning trace, "
    "the candidate final answer, and the strict verifier's repair instruction. "
    "Repair only format, completeness, slot count, precision, and requested answer form. "
    "Do not call tools. Return exactly one final boxed answer and no prose."
)

RecordSink = Callable[[dict[str, Any]], None]
VerifierResult = dict[str, Any]

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

ERROR_NONE = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify and repair finalized Qwen answers.")
    parser.add_argument("--data", default="data/public.jsonl", help="Competition data JSONL.")
    parser.add_argument("--raw-responses", required=True, help="Raw 8k Qwen JSONL.")
    parser.add_argument("--finalized-responses", required=True, help="LoRA-finalized JSONL.")
    parser.add_argument("--output", default="codex_added/results/verify_repair.jsonl")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--max-trace-chars", type=int, default=6000)
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--repair-all",
        action="store_true",
        help="Call Qwen for every row, including rows that pass the strict verifier.",
    )
    parser.add_argument(
        "--only-repair-bad-format",
        action="store_true",
        help="Deprecated compatibility flag. Strict repair-only behavior is now the default.",
    )
    parser.add_argument("--score", action="store_true")
    add_vllm_cli_args(parser)
    return parser.parse_args()


def problem_text(item: dict[str, Any]) -> str:
    text = str(item["question"])
    if item.get("options"):
        text += "\n\nOptions:\n" + format_options(item["options"])
    return text


def expected_slot_count(item: dict[str, Any]) -> int:
    """Count answer slots from the problem text, not from the gold answer."""
    if is_mcq(item):
        return 1
    return str(item.get("question", "")).count("[ANS]") or 1


def strict_final_format_contract(item: dict[str, Any]) -> str:
    if is_mcq(item):
        option_count = len(item.get("options") or [])
        letters = ", ".join(chr(65 + index) for index in range(option_count))
        return (
            "Output exactly one line: \\boxed{X}, where X is exactly one option "
            f"letter from {{{letters}}}. Do not include option text or prose."
        )

    slots = expected_slot_count(item)
    plural = "answer" if slots == 1 else "ordered answers"
    return (
        f"Output exactly one line: \\boxed{{...}} containing exactly {slots} {plural}. "
        "Use one comma-separated entry per [ANS] blank, in order. Match requested "
        "rounding, precision, percent format, decimal form, fraction form, or exact "
        "symbolic form."
    )


def _result(
    *,
    valid: bool,
    error_type: str,
    expected_slots: int,
    predicted_slots: int,
    repair_instruction: str,
) -> VerifierResult:
    return {
        "valid": valid,
        "needs_repair": not valid,
        "error_type": error_type,
        "expected_slots": int(expected_slots),
        "predicted_slots": int(predicted_slots),
        "repair_instruction": repair_instruction,
    }


def boxed_spans(response: str) -> tuple[list[tuple[int, int, str]], bool]:
    spans: list[tuple[int, int, str]] = []
    malformed = False
    index = 0
    while True:
        start = response.find("\\boxed", index)
        if start < 0:
            break
        brace = start + len("\\boxed")
        while brace < len(response) and response[brace].isspace():
            brace += 1
        if brace >= len(response) or response[brace] != "{":
            malformed = True
            index = start + len("\\boxed")
            continue

        depth = 0
        end = -1
        for pos in range(brace, len(response)):
            char = response[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = pos
                    break
        if end < 0:
            malformed = True
            spans.append((start, len(response), response[brace + 1 :]))
            break
        spans.append((start, end + 1, response[brace + 1 : end]))
        index = end + 1
    return spans, malformed


def split_answer_slots(content: str) -> list[str]:
    slots: list[str] = []
    start = 0
    depth = 0
    pairs = {"{": "}", "(": ")", "[": "]"}
    closers = set(pairs.values())
    for index, char in enumerate(content):
        if char in pairs:
            depth += 1
        elif char in closers and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            slots.append(content[start:index].strip())
            start = index + 1
    tail = content[start:].strip()
    if tail or content:
        slots.append(tail)
    return slots


def strip_answer_wrappers(value: str) -> str:
    value = str(value).strip().strip("$").strip()
    if value.startswith("\\(") and value.endswith("\\)"):
        value = value[2:-2].strip()
    if value.startswith("\\[") and value.endswith("\\]"):
        value = value[2:-2].strip()
    return value


def _number_word_or_digit_to_int(value: str) -> int | None:
    value = value.strip().lower()
    if value.isdigit():
        return int(value)
    return NUMBER_WORDS.get(value)


def requested_answer_policy(item: dict[str, Any]) -> dict[str, Any]:
    question = str(item.get("question", "")).lower()
    allow_decimal_or_fraction = bool(
        re.search(r"\b(?:decimal\s+or\s+fraction|fraction\s+or\s+decimal)\b", question)
    )

    decimal_places: int | None = None
    place_patterns = [
        (r"nearest\s+(?:whole\s+number|integer)", 0),
        (r"nearest\s+tenth\b", 1),
        (r"nearest\s+hundredth\b", 2),
        (r"nearest\s+thousandth\b", 3),
        (r"nearest\s+ten[-\s]?thousandth\b", 4),
        (r"nearest\s+cent\b", 2),
        (r"hundredths?\s+place", 2),
        (r"tenths?\s+place", 1),
        (r"thousandths?\s+place", 3),
    ]
    for pattern, places in place_patterns:
        if re.search(pattern, question):
            decimal_places = places
            break

    if decimal_places is None:
        match = re.search(
            r"(?:to|with|rounded\s+to)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+decimal\s+places?",
            question,
        )
        if not match:
            match = re.search(
                r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+places?\s+after\s+the\s+decimal",
                question,
            )
        if match:
            decimal_places = _number_word_or_digit_to_int(match.group(1))

    fraction_required = bool(
        re.search(
            r"\b(?:as\s+a\s+fraction|common\s+fraction|mixed\s+number|improper\s+fraction|"
            r"fraction\s+form|fractional\s+form|reduced\s+fraction|"
            r"simpl(?:ify|ified)\s+(?:the\s+)?fraction|lowest\s+terms|simplest\s+form)\b",
            question,
        )
        or (
            re.search(r"\b(?:reduce|simplify|simplified)\b", question)
            and (re.search(r"\d+\s*/\s*\d+", question) or "\\frac" in question)
        )
    ) and not allow_decimal_or_fraction
    decimal_required = bool(
        decimal_places is not None
        or (re.search(r"\bdecimal\b", question) and not allow_decimal_or_fraction)
    )
    percent_required = bool(re.search(r"\b(?:as\s+a\s+percent|percentage|percent\s+form)\b|%", question))
    exact_preferred = bool(
        fraction_required
        or (
            re.search(
                r"\b(?:exact|exact\s+form|in\s+terms\s+of|radical\s+form|simplified\s+radical|"
                r"symbolic\s+form)\b|\\pi|π",
                question,
            )
            and not decimal_required
        )
    )
    return {
        "allow_decimal_or_fraction": allow_decimal_or_fraction,
        "decimal_places": decimal_places,
        "fraction_required": fraction_required,
        "decimal_required": decimal_required,
        "percent_required": percent_required,
        "exact_preferred": exact_preferred,
    }


def contains_decimal(value: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z])[-+]?\d[\d,]*\.\d+", value))


def contains_fraction(value: str) -> bool:
    return "\\frac" in value or bool(re.search(r"[-+]?\d[\d,]*\s*/\s*[-+]?\d[\d,]*", value))


def contains_exact_symbolic(value: str) -> bool:
    return bool(re.search(r"\\sqrt|sqrt|\\pi|\bpi\b|π|\^", value, re.IGNORECASE))


def decimal_place_count(value: str) -> int | None:
    clean = strip_answer_wrappers(value).replace(",", "")
    match = re.fullmatch(r"[-+]?\d+\.(\d+)%?", clean)
    if not match:
        return None
    return len(match.group(1))


def numeric_value(value: str) -> float | None:
    clean = strip_answer_wrappers(value).replace(",", "").rstrip("%")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", clean):
        return None
    try:
        return float(clean)
    except ValueError:
        return None


def is_integer_form(value: str) -> bool:
    clean = strip_answer_wrappers(value).replace(",", "").rstrip("%")
    return bool(re.fullmatch(r"[-+]?\d+", clean))


def malformed_slot(value: str) -> bool:
    clean = strip_answer_wrappers(value)
    if not clean:
        return True
    bad_markers = ("[ANS]", "</think>", "<|", "...", "???")
    return any(marker in clean for marker in bad_markers)


def requested_form_error(item: dict[str, Any], slots: list[str]) -> tuple[str, str] | None:
    policy = requested_answer_policy(item)
    places = policy["decimal_places"]
    for slot in slots:
        clean = strip_answer_wrappers(slot)
        if places == 0:
            if contains_decimal(clean) or contains_fraction(clean):
                return (
                    "precision_mismatch",
                    "Round to the requested nearest integer or whole number and output an integer.",
                )
        elif places is not None:
            slot_places = decimal_place_count(clean)
            if slot_places != places:
                return (
                    "precision_mismatch",
                    f"Round to exactly {places} decimal place(s) and keep trailing zeros if needed.",
                )

        if policy["fraction_required"] and contains_decimal(clean) and not contains_fraction(clean):
            return (
                "exact_form_mismatch",
                "Convert the decimal to the requested exact reduced fraction form.",
            )

        if policy["decimal_required"] and not policy["allow_decimal_or_fraction"]:
            if contains_fraction(clean) or (contains_exact_symbolic(clean) and not contains_decimal(clean)):
                return (
                    "exact_form_mismatch",
                    "Convert the answer to the requested decimal form and requested precision.",
                )

        if policy["exact_preferred"] and contains_decimal(clean) and not policy["allow_decimal_or_fraction"]:
            return (
                "exact_form_mismatch",
                "Use the requested exact form instead of a decimal approximation.",
            )

        if policy["percent_required"]:
            value = numeric_value(clean)
            if contains_fraction(clean):
                return (
                    "exact_form_mismatch",
                    "Convert the fraction to the requested percent form.",
                )
            if value is not None and 0 < abs(value) < 1 and "%" not in clean:
                return (
                    "precision_mismatch",
                    "Convert the probability-scale decimal to the requested percent scale.",
                )
    return None


def strict_verify_final_answer(item: dict[str, Any], response: str) -> VerifierResult:
    expected_slots = expected_slot_count(item)
    response = str(response).strip()
    spans, malformed_box = boxed_spans(response)
    if not spans:
        return _result(
            valid=False,
            error_type="missing_box",
            expected_slots=expected_slots,
            predicted_slots=0,
            repair_instruction="Return exactly one final answer in \\boxed{...}.",
        )
    if len(spans) > 1:
        return _result(
            valid=False,
            error_type="multiple_boxes",
            expected_slots=expected_slots,
            predicted_slots=len(spans),
            repair_instruction="Combine the final answer into one \\boxed{...} and remove all other boxes.",
        )
    if malformed_box:
        return _result(
            valid=False,
            error_type="malformed_answer",
            expected_slots=expected_slots,
            predicted_slots=0,
            repair_instruction="Repair the malformed boxed answer and return one complete \\boxed{...}.",
        )

    start, end, content = spans[0]
    outside = (response[:start] + response[end:]).strip()
    if outside:
        return _result(
            valid=False,
            error_type="malformed_answer",
            expected_slots=expected_slots,
            predicted_slots=expected_slots,
            repair_instruction="Remove all explanation text and return only one \\boxed{...}.",
        )

    content = content.strip()
    if is_mcq(item):
        option_count = len(item.get("options") or [])
        allowed = {chr(65 + index) for index in range(option_count)}
        letter = content.upper()
        predicted_slots = 1 if content else 0
        if letter not in allowed or not re.fullmatch(r"[A-Z]", letter):
            return _result(
                valid=False,
                error_type="mcq_invalid_letter",
                expected_slots=1,
                predicted_slots=predicted_slots,
                repair_instruction="Return exactly one valid multiple-choice option letter inside \\boxed{}.",
            )
        return _result(
            valid=True,
            error_type=ERROR_NONE,
            expected_slots=1,
            predicted_slots=1,
            repair_instruction="",
        )

    slots = split_answer_slots(content)
    predicted_slots = len(slots) if content else 0
    if predicted_slots != expected_slots:
        error_type = "truncated_or_incomplete" if expected_slots > 1 and predicted_slots <= 1 else "wrong_slot_count"
        return _result(
            valid=False,
            error_type=error_type,
            expected_slots=expected_slots,
            predicted_slots=predicted_slots,
            repair_instruction=(
                f"Return exactly {expected_slots} ordered comma-separated answer(s), one for each [ANS] blank."
            ),
        )
    if any(malformed_slot(slot) for slot in slots):
        return _result(
            valid=False,
            error_type="truncated_or_incomplete",
            expected_slots=expected_slots,
            predicted_slots=predicted_slots,
            repair_instruction="Fill every slot with a complete answer and remove placeholders or truncated text.",
        )

    form_error = requested_form_error(item, slots)
    if form_error:
        error_type, instruction = form_error
        return _result(
            valid=False,
            error_type=error_type,
            expected_slots=expected_slots,
            predicted_slots=predicted_slots,
            repair_instruction=instruction,
        )

    return _result(
        valid=True,
        error_type=ERROR_NONE,
        expected_slots=expected_slots,
        predicted_slots=predicted_slots,
        repair_instruction="",
    )


def verification_rules(item: dict[str, Any]) -> str:
    if is_mcq(item):
        letters = ", ".join(chr(65 + index) for index, _ in enumerate(item.get("options") or []))
        return "\n".join(
            [
                f"- This is multiple choice; the final answer must be one letter from {{{letters}}}.",
                "- Check that the letter matches the option text, not just an equivalent-looking expression.",
                "- If the candidate uses option text instead of a letter, repair it to the matching letter.",
            ]
        )
    slots = expected_slot_count(item)
    return "\n".join(
        [
            f"- This is free-form; the problem has {slots} [ANS] slot(s).",
            f"- The final box must contain exactly {slots} ordered comma-separated field(s).",
            "- If the candidate has too few or too many fields, repair it from the reasoning trace.",
            "- Preserve exact symbolic forms and requested precision from the trace.",
        ]
    )


def build_prompt(
    tokenizer: Any,
    item: dict[str, Any],
    raw_response: str,
    finalized_response: str,
    verification: VerifierResult,
    max_trace_chars: int,
) -> str:
    trace = raw_response[-max_trace_chars:]
    messages = [
        {"role": "system", "content": REPAIR_SYSTEM},
        {
            "role": "user",
            "content": (
                "Problem:\n"
                f"{problem_text(item)}\n\n"
                "Required final-answer schema:\n"
                f"{strict_final_format_contract(item)}\n\n"
                "Verification and repair rules:\n"
                f"{verification_rules(item)}\n\n"
                "Strict verifier JSON:\n"
                f"{json.dumps(verification, ensure_ascii=False, sort_keys=True)}\n\n"
                "Candidate final answer from the LoRA finalizer:\n"
                f"{finalized_response}\n\n"
                "Previous raw Qwen reasoning trace:\n"
                f"{trace}\n\n"
                "Repair only the verifier issue. End with the final answer only in one box."
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_transformers_model(model_id: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def make_record(
    item: dict[str, Any],
    raw_row: dict[str, Any],
    finalized_row: dict[str, Any],
    verifier_response: str,
    generated_tokens: int | None,
    repaired_by: str,
    verification: VerifierResult,
    post_repair_verification: VerifierResult | None = None,
) -> dict[str, Any]:
    normalized = normalize_final_response(item, verifier_response)
    final_verification = post_repair_verification or verification
    record = {
        "id": int(item["id"]),
        "is_mcq": is_mcq(item),
        "raw_response": str(raw_row.get("response", "")),
        "finalizer_response": str(finalized_row.get("response", "")),
        "verifier_response": verifier_response,
        "response": normalized,
        "answer_key": extract_answer_key(item, normalized, strict=True),
        "repaired_by": repaired_by,
        "verification": verification,
        "strict_valid": bool(final_verification["valid"]),
        "error_type": str(final_verification["error_type"]),
        "expected_slots": int(final_verification["expected_slots"]),
        "parsed_slots": int(final_verification["predicted_slots"]),
        "format_ok": bool(final_verification["valid"]),
    }
    if post_repair_verification is not None:
        record["post_repair_verification"] = post_repair_verification
    if generated_tokens is not None:
        record["generated_tokens"] = generated_tokens
    if "diagnostic_group" in item:
        record["diagnostic_group"] = item["diagnostic_group"]
    return record


def needs_repair(item: dict[str, Any], finalized_row: dict[str, Any]) -> bool:
    return not bool(strict_verify_final_answer(item, str(finalized_row.get("response", "")))["valid"])


def verify_with_transformers(
    items: list[dict[str, Any]],
    raw_by_id: dict[int, dict[str, Any]],
    final_by_id: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    emit_record: RecordSink | None = None,
) -> list[dict[str, Any]]:
    import torch

    records: list[dict[str, Any]] = []
    repair_jobs: list[tuple[dict[str, Any], VerifierResult]] = []
    for item in items:
        item_id = int(item["id"])
        raw_row = raw_by_id[item_id]
        final_row = final_by_id[item_id]
        verification = strict_verify_final_answer(item, str(final_row.get("response", "")))
        if verification["valid"] and not args.repair_all:
            record = make_record(
                item,
                raw_row,
                final_row,
                str(final_row.get("response", "")),
                0,
                "kept_strict_valid",
                verification,
            )
            records.append(record)
            if emit_record:
                emit_record(record)
            continue
        repair_jobs.append((item, verification))

    if not repair_jobs:
        return records

    tokenizer, model = load_transformers_model(args.model_id)
    for item, verification in tqdm(repair_jobs, desc="Verifying/repairing"):
        item_id = int(item["id"])
        raw_row = raw_by_id[item_id]
        final_row = final_by_id[item_id]
        prompt = build_prompt(
            tokenizer,
            item,
            str(raw_row.get("response", "")),
            str(final_row.get("response", "")),
            verification,
            args.max_trace_chars,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        verifier_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        normalized = normalize_final_response(item, verifier_response)
        post_verification = strict_verify_final_answer(item, normalized)
        record = make_record(
            item,
            raw_row,
            final_row,
            verifier_response,
            int(new_tokens.shape[0]),
            "qwen_verify_repair",
            verification,
            post_verification,
        )
        records.append(record)
        if emit_record:
            emit_record(record)
    return records


def verify_with_vllm(
    items: list[dict[str, Any]],
    raw_by_id: dict[int, dict[str, Any]],
    final_by_id: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    emit_record: RecordSink | None = None,
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    records: list[dict[str, Any]] = []
    repair_jobs: list[tuple[dict[str, Any], VerifierResult]] = []
    for item in items:
        item_id = int(item["id"])
        raw_row = raw_by_id[item_id]
        final_row = final_by_id[item_id]
        verification = strict_verify_final_answer(item, str(final_row.get("response", "")))
        if verification["valid"] and not args.repair_all:
            record = make_record(
                item,
                raw_row,
                final_row,
                str(final_row.get("response", "")),
                0,
                "kept_strict_valid",
                verification,
            )
            records.append(record)
            if emit_record:
                emit_record(record)
            continue
        repair_jobs.append((item, verification))

    if not repair_jobs:
        records.sort(key=lambda row: int(row["id"]))
        return records

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = make_llm(args.model_id, args, trust_remote_code=True)
    sampling_params = make_sampling_params(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )
    chunk_size = int(args.vllm_batch_size or len(repair_jobs) or 1)
    for batch in tqdm(list(batched(repair_jobs, chunk_size)), desc="Verifying/repairing"):
        prompts: list[str] = []
        prompt_items: list[dict[str, Any]] = []
        prompt_verifications: list[VerifierResult] = []
        for item, verification in batch:
            item_id = int(item["id"])
            raw_row = raw_by_id[item_id]
            final_row = final_by_id[item_id]
            prompts.append(
                build_prompt(
                    tokenizer,
                    item,
                    str(raw_row.get("response", "")),
                    str(final_row.get("response", "")),
                    verification,
                    args.max_trace_chars,
                )
            )
            prompt_items.append(item)
            prompt_verifications.append(verification)
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        for item, verification, output in zip(prompt_items, prompt_verifications, outputs):
            item_id = int(item["id"])
            verifier_response = vllm_text(output)
            normalized = normalize_final_response(item, verifier_response)
            post_verification = strict_verify_final_answer(item, normalized)
            record = make_record(
                item,
                raw_by_id[item_id],
                final_by_id[item_id],
                verifier_response,
                vllm_generated_tokens(output),
                "qwen_verify_repair",
                verification,
                post_verification,
            )
            records.append(record)
            if emit_record:
                emit_record(record)
    records.sort(key=lambda row: int(row["id"]))
    return records


def score_record_in_place(
    record: dict[str, Any],
    data_by_id: dict[int, dict[str, Any]],
    judger: Judger | None,
) -> None:
    if judger is None:
        return
    item = data_by_id[int(record["id"])]
    if has_gold(item):
        record["gold"] = item["answer"]
        record["correct"] = score_item(judger, item, str(record.get("response", "")))


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    data_by_id = index_by_id(data)
    raw_by_id = index_by_id(read_jsonl(args.raw_responses))
    final_by_id = index_by_id(read_jsonl(args.finalized_responses))
    output_path = Path(args.output)
    existing: list[dict[str, Any]] = read_jsonl_complete_prefix(output_path) if output_path.exists() else []
    seen_ids = {int(record["id"]) for record in existing}
    judger = Judger(strict_extract=False) if args.score else None

    def emit_record(record: dict[str, Any]) -> None:
        score_record_in_place(record, data_by_id, judger)
        append_jsonl_record(output_path, record)

    data = [
        item
        for item in data
        if int(item["id"]) in raw_by_id and int(item["id"]) in final_by_id and int(item["id"]) not in seen_ids
    ]
    if existing:
        print(f"Resuming from {len(existing)} existing verifier/repair rows in {output_path}.")

    if args.backend == "vllm":
        records = verify_with_vllm(data, raw_by_id, final_by_id, args, emit_record)
    else:
        records = verify_with_transformers(data, raw_by_id, final_by_id, args, emit_record)

    records = existing + records
    if args.score:
        for record in records:
            score_record_in_place(record, data_by_id, judger)
        print(json.dumps(summarize_results(records), indent=2, sort_keys=True))

    records.sort(key=lambda record: int(record["id"]))
    write_jsonl(args.output, records)
    print(f"Wrote verifier/repair responses to {args.output}")


if __name__ == "__main__":
    main()
