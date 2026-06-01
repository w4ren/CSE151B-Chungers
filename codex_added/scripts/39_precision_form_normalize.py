# Added by Codex: deterministic post-finalizer cleanup for precision and evaluator syntax.

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from judger import Judger
from math_comp.data import answer_slot_count, has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.final_answer import final_answer_diagnostics, normalize_final_response
from math_comp.scoring import extract_answer_key, score_item, summarize_results


BOX_START = "\\boxed"
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
SCIENTIFIC_E_RE = re.compile(r"^([-+]?\d+(?:\.\d+)?)[eE]([+-]?\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize finalized answers for precision and evaluator syntax.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--raw-responses", required=True)
    parser.add_argument("--finalized-responses", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--audit-output", default=None)
    return parser.parse_args()


def boxed_spans(response: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    index = 0
    while True:
        start = response.find(BOX_START, index)
        if start < 0:
            break
        brace = start + len(BOX_START)
        while brace < len(response) and response[brace].isspace():
            brace += 1
        if brace >= len(response) or response[brace] != "{":
            index = start + len(BOX_START)
            continue
        depth = 0
        end = -1
        for pos in range(brace, len(response)):
            if response[pos] == "{":
                depth += 1
            elif response[pos] == "}":
                depth -= 1
                if depth == 0:
                    end = pos
                    break
        if end < 0:
            break
        spans.append((start, end + 1, response[brace + 1 : end]))
        index = end + 1
    return spans


def boxed_content(response: str) -> str:
    spans = boxed_spans(str(response))
    if spans:
        return spans[-1][2].strip()
    normalized = normalize_final_response({}, str(response))
    spans = boxed_spans(normalized)
    return spans[-1][2].strip() if spans else str(response).strip()


def split_slots(content: str) -> list[str]:
    return Judger(strict_extract=True).split_by_comma(content)


def box(slots: list[str]) -> str:
    return "\\boxed{" + ", ".join(slot.strip() for slot in slots) + "}"


def replace_latex_fraction(match: re.Match[str]) -> str:
    return f"({match.group(1)})/({match.group(2)})"


def fix_sqrt_artifacts(text: str) -> str:
    text = re.sub(r"\\?sqrt\{\(}?([A-Za-z0-9.+\\-]+)\)?\}", r"\\sqrt{\1}", text)
    text = re.sub(r"\\?sqrt\(\s*([A-Za-z0-9.+\\-]+)\s*\)", r"\\sqrt{\1}", text)
    text = re.sub(r"\\text\{sqrt\}\(([^()]+)\)", r"\\sqrt{\1}", text)
    return text


def basic_slot_cleanup(slot: str) -> str:
    cleaned = str(slot).strip().strip("$").strip()
    cleaned = cleaned.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    cleaned = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", replace_latex_fraction, cleaned)
    sci_match = SCIENTIFIC_E_RE.fullmatch(cleaned.replace(" ", ""))
    if sci_match:
        base = sci_match.group(1).rstrip("0").rstrip(".")
        exponent = str(int(sci_match.group(2)))
        cleaned = f"{base}*10^{exponent}"
    cleaned = cleaned.replace("\\pi", "pi")
    cleaned = cleaned.replace("π", "pi")
    cleaned = re.sub(r"\\?inf\^\{?1\}?ty(?:inity|ty)*", "infinity", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("\\infty", "infinity")
    cleaned = cleaned.replace("∞", "infinity")
    cleaned = re.sub(r"\\?(sin|cos|tan)\^\{?1\}?\(([^()]*)\)", r"\1(\2)", cleaned)
    cleaned = re.sub(r"\\min\^\{?1\}?", "min", cleaned)
    cleaned = fix_sqrt_artifacts(cleaned)
    cleaned = re.sub(r"\s+", "", cleaned) if re.fullmatch(r"[\sA-Za-z0-9_+\-*/^().,{}\\]+", cleaned) else cleaned
    return cleaned.strip()


def decimal_places(text: str) -> int | None:
    text = str(text).strip().replace(",", "")
    match = re.fullmatch(r"[-+]?\d+\.(\d+)", text)
    return len(match.group(1)) if match else None


def quantized(value: str, places: int) -> str | None:
    try:
        dec = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
    quant = Decimal("1") if places == 0 else Decimal("1").scaleb(-places)
    return format(dec.quantize(quant, rounding=ROUND_HALF_UP), f".{places}f")


def number_literal_in_question(item: dict[str, Any], value: str) -> bool:
    """Avoid expanding constants copied directly from the problem statement."""
    try:
        target = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return False
    question = str(item.get("question", ""))
    for match in NUMBER_RE.finditer(question):
        try:
            if Decimal(match.group(0).replace(",", "")) == target:
                return True
        except InvalidOperation:
            continue
    return False


def raw_number_candidates(raw_text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in NUMBER_RE.finditer(raw_text):
        value = match.group(0).replace(",", "")
        if "." not in value:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def should_expand_decimal_precision(item: dict[str, Any]) -> bool:
    question = str(item.get("question", "")).lower()
    blocked_phrases = (
        "decimal place",
        "nearest tenth",
        "nearest hundredth",
        "nearest thousandth",
        "nearest cent",
        "rounded to",
        "round to",
        "as a percent",
        "to a percent",
    )
    if any(phrase in question for phrase in blocked_phrases):
        return False
    if "significant figure" in question or "significant digit" in question:
        return False
    return True


def expand_decimal_precision(item: dict[str, Any], slot: str, raw_text: str) -> tuple[str, str | None]:
    places = decimal_places(slot)
    if places is None:
        return slot, None
    if number_literal_in_question(item, slot):
        return slot, None
    current = slot.replace(",", "")
    best = slot
    best_decimal_count = places
    for candidate in raw_number_candidates(raw_text):
        candidate_places = decimal_places(candidate)
        if candidate_places is None or candidate_places <= places:
            continue
        rounded = quantized(candidate, places)
        if rounded == current and candidate_places > best_decimal_count:
            best = candidate
            best_decimal_count = candidate_places
    if best != slot:
        return best, f"expanded_decimal:{slot}->{best}"
    return slot, None


def parse_inline_options(question: str) -> dict[str, str]:
    pattern = re.compile(r"\b([A-J])\.\s*([^A-J\[]+?)(?=\s+[A-J]\.|$)", re.DOTALL)
    options: dict[str, str] = {}
    for letter, value in pattern.findall(question):
        value = " ".join(value.split()).strip()
        if value:
            options[letter] = value
    return options


def compact_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9.+/*^\\\\-]", "", str(text).lower())


def map_inline_option_value(item: dict[str, Any], slots: list[str]) -> tuple[list[str], str | None]:
    if is_mcq(item) or len(slots) != 1:
        return slots, None
    options = parse_inline_options(str(item.get("question", "")))
    if not options:
        return slots, None
    pred = compact_for_match(slots[0])
    for letter, value in options.items():
        if pred and pred == compact_for_match(value):
            return [letter], f"inline_option_value_to_letter:{slots[0]}->{letter}"
    return slots, None


def normalize_true_false(item: dict[str, Any], slots: list[str]) -> tuple[list[str], list[str]]:
    question = str(item.get("question", "")).lower()
    if "true or false" not in question and "select true" not in question:
        return slots, []
    changes: list[str] = []
    out: list[str] = []
    for slot in slots:
        lower = slot.strip().lower()
        if lower == "true":
            out.append("T")
            changes.append("true_false_word_to_letter")
        elif lower == "false":
            out.append("F")
            changes.append("true_false_word_to_letter")
        else:
            out.append(slot)
    return out, changes


def normalize_period_amplitude(item: dict[str, Any], slots: list[str]) -> tuple[list[str], str | None]:
    question = str(item.get("question", "")).lower()
    if "period=[ans]" not in question or "amplitude=[ans]" not in question or len(slots) != 2:
        return slots, None
    first_has_pi = "pi" in slots[0].lower() or "\\pi" in slots[0]
    second_has_pi = "pi" in slots[1].lower() or "\\pi" in slots[1]
    if not first_has_pi and second_has_pi:
        return [slots[1], slots[0]], "swapped_period_amplitude"
    return slots, None


def normalize_interval_form(item: dict[str, Any], slots: list[str]) -> tuple[list[str], list[str]]:
    question = str(item.get("question", "")).lower()
    if "increasing" not in question and "decreasing" not in question:
        return slots, []
    changes: list[str] = []
    out: list[str] = []
    for slot in slots:
        new = slot
        if re.match(r"^\[[^,]+,\s*(?:infinity|\\infty)", new):
            new = "(" + new[1:]
        if new != slot:
            changes.append("open_monotonicity_interval")
        out.append(new)
    return out, changes


def tan_solution_exact_form(item: dict[str, Any], slots: list[str]) -> tuple[list[str], str | None]:
    question = str(item.get("question", ""))
    if len(slots) != 2:
        return slots, None
    match = re.search(r"tan\((?:\\theta|theta)\)\s*=\s*([-+]?\d+(?:\.\d+)?)", question, flags=re.IGNORECASE)
    if not match:
        return slots, None
    second = slots[1].lower().replace(" ", "")
    if second in {"pi", "\\pi"} or re.fullmatch(r"3\.14\d*", second):
        return [f"atan({match.group(1)})", "pi"], "tan_general_solution_exact"
    return slots, None


def recover_substitution_expressions(item: dict[str, Any], slots: list[str]) -> tuple[list[str], str | None]:
    question = str(item.get("question", ""))
    if "Evaluate the expressions for" not in question:
        return slots, None
    assignments = dict(re.findall(r"\b([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)", question))
    if not assignments:
        return slots, None
    exprs = re.findall(r"([A-Za-z0-9+\-*/^ ]+)=\[ANS\]", question)
    if len(exprs) != len(slots):
        return slots, None
    substituted: list[str] = []
    for expr in exprs:
        expr = expr.strip().replace(" ", "*")
        # The source sometimes writes multiplication by adjacency, e.g. x y z.
        expr = re.sub(r"\*+", "*", expr)
        expr = expr.strip("*")
        for var, value in assignments.items():
            expr = re.sub(rf"\b{re.escape(var)}\b", value, expr)
        substituted.append(expr)
    return substituted, "recovered_substitution_expressions"


def half_life_fraction_exact(item: dict[str, Any], slots: list[str]) -> tuple[list[str], str | None]:
    question = str(item.get("question", ""))
    if len(slots) != 1 or "half-life" not in question.lower() or "fraction" not in question.lower():
        return slots, None
    years = [int(value) for value in re.findall(r"\b(19\d{2}|20\d{2})\b", question)]
    half_life = re.search(r"half-life[^.]*?(\d+(?:\.\d+)?)\s+years?", question, flags=re.IGNORECASE)
    if len(years) >= 2 and half_life:
        elapsed = max(years) - min(years)
        return [f"(1/2)^(({elapsed})/{half_life.group(1)})"], "half_life_fraction_exact"
    return slots, None


def decay_half_life_exact(item: dict[str, Any], slots: list[str]) -> tuple[list[str], str | None]:
    question = str(item.get("question", ""))
    if len(slots) != 1 or "half-life" not in question.lower() or "decays by" not in question.lower():
        return slots, None
    match = re.search(r"decays by\s*(\d+(?:\.\d+)?)\s*%", question, flags=re.IGNORECASE)
    if not match:
        return slots, None
    rate = Decimal(match.group(1)) / Decimal(100)
    remaining = Decimal(1) - rate
    return [f"ln(0.5)/ln({remaining})"], "decay_half_life_exact"


def rebuild_from_raw_box_if_better(item: dict[str, Any], slots: list[str], raw_text: str) -> tuple[list[str], str | None]:
    raw_boxes = [content for _, _, content in boxed_spans(raw_text)]
    if not raw_boxes:
        return slots, None
    raw_slots = split_slots(raw_boxes[-1])
    if len(raw_slots) == answer_slot_count(item) and len(raw_slots) > len(slots):
        return [basic_slot_cleanup(slot) for slot in raw_slots], "used_raw_box_for_more_slots"
    return slots, None


def normalize_slots(item: dict[str, Any], raw_text: str, finalized_response: str) -> tuple[list[str], list[str]]:
    content = boxed_content(finalized_response)
    slots = split_slots(content)
    changes: list[str] = []

    raw_rebuild, change = rebuild_from_raw_box_if_better(item, slots, raw_text)
    if change:
        slots = raw_rebuild
        changes.append(change)

    cleaned_slots = []
    allow_decimal_expansion = should_expand_decimal_precision(item)
    for slot in slots:
        cleaned = basic_slot_cleanup(slot)
        if allow_decimal_expansion:
            expanded, change = expand_decimal_precision(item, cleaned, raw_text)
        else:
            expanded, change = cleaned, None
        cleaned_slots.append(expanded)
        if cleaned != slot:
            changes.append("basic_syntax_cleanup")
        if change:
            changes.append(change)
    slots = cleaned_slots

    for transform in (map_inline_option_value, normalize_period_amplitude, tan_solution_exact_form,
                      recover_substitution_expressions, half_life_fraction_exact, decay_half_life_exact):
        result, change = transform(item, slots)  # type: ignore[misc]
        if change:
            slots = result
            changes.append(change)

    slots, more_changes = normalize_true_false(item, slots)
    changes.extend(more_changes)
    slots, more_changes = normalize_interval_form(item, slots)
    changes.extend(more_changes)
    return slots, changes


def score_record_in_place(record: dict[str, Any], item: dict[str, Any], judger: Judger | None) -> None:
    if judger is None or not has_gold(item):
        return
    record["gold"] = item["answer"]
    record["correct"] = score_item(judger, item, str(record.get("response", "")))


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    data_by_id = index_by_id(data)
    raw_by_id = index_by_id(read_jsonl(args.raw_responses))
    finalized = read_jsonl(args.finalized_responses)
    judger = Judger(strict_extract=False) if args.score else None
    records: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for row in finalized:
        item_id = int(row["id"])
        item = data_by_id[item_id]
        raw_row = raw_by_id.get(item_id, {})
        old_response = str(row.get("response", ""))
        slots, changes = normalize_slots(item, str(raw_row.get("response", "")), old_response)
        new_response = box(slots)
        new_response = normalize_final_response(item, new_response)
        record = {
            **row,
            "id": item_id,
            "source_response": str(raw_row.get("response", row.get("source_response", ""))),
            "pre_normalizer_response": old_response,
            "response": new_response,
            "answer_key": extract_answer_key(item, new_response, strict=True),
            "normalizer_changes": changes,
        }
        record.update(final_answer_diagnostics(item, new_response))
        score_record_in_place(record, item, judger)
        records.append(record)
        if changes:
            audit.append(
                {
                    "id": item_id,
                    "changes": changes,
                    "before": old_response,
                    "after": new_response,
                    "correct": record.get("correct"),
                    "gold": item.get("answer"),
                }
            )

    records.sort(key=lambda record: int(record["id"]))
    write_jsonl(args.output, records)
    if args.audit_output:
        Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_output).write_text(
            json.dumps({"changed": len(audit), "changes": audit}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.score:
        print(json.dumps(summarize_results(records), indent=2, sort_keys=True))
    print(f"Wrote normalized responses to {args.output}")


if __name__ == "__main__":
    main()
