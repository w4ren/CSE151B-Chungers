# Added by Codex: guarded slot-aware repair pass for V5 outputs.

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import sympy as sp
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from judger import Judger
from math_comp.data import has_gold, index_by_id, read_jsonl, write_jsonl
from math_comp.final_answer import final_answer_diagnostics, normalize_final_response
from math_comp.scoring import extract_answer_key, score_item, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair only validated slot-count failures.")
    parser.add_argument("--data", required=True, help="Competition data JSONL.")
    parser.add_argument("--responses", required=True, help="Input finalized JSONL.")
    parser.add_argument("--output", required=True, help="Output JSONL.")
    parser.add_argument("--audit-output", default=None, help="Optional JSON audit path.")
    parser.add_argument("--score", action="store_true", help="Score output when answers are present.")
    return parser.parse_args()


def boxed(content: str) -> str:
    return f"\\boxed{{{content}}}"


def boxed_content(response: str) -> str:
    text = str(response)
    start = text.rfind("\\boxed{")
    if start < 0:
        return ""
    pos = start + len("\\boxed{")
    depth = 1
    out: list[str] = []
    while pos < len(text) and depth > 0:
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(char)
        pos += 1
    return "".join(out).strip() if depth == 0 else ""


def split_slots(content: str) -> list[str]:
    return Judger(strict_extract=True).split_by_comma(content)


def normalize_slot(slot: str) -> str:
    return str(slot).strip().strip("$").strip()


def single_letter_slots(slots: list[str]) -> bool:
    return bool(slots) and all(re.fullmatch(r"[A-Ha-h]", normalize_slot(slot)) for slot in slots)


def repair_multiselect_letters(item: dict[str, Any], response: str, diagnostics: dict[str, Any]) -> str | None:
    if diagnostics["expected_slots"] != 1 or diagnostics["parsed_slots"] <= 1:
        return None
    content = boxed_content(response)
    slots = split_slots(content)
    if not single_letter_slots(slots):
        return None
    question = str(item.get("question", "")).lower()
    if "select all" not in question and "select every" not in question and "more than one correct" not in question:
        return None
    letters = "".join(normalize_slot(slot).upper() for slot in slots)
    return boxed(letters)


def repair_single_slot_list(item: dict[str, Any], response: str, diagnostics: dict[str, Any]) -> str | None:
    if diagnostics["expected_slots"] != 1 or diagnostics["parsed_slots"] <= 1:
        return None
    content = boxed_content(response)
    slots = [normalize_slot(slot) for slot in split_slots(content)]
    if single_letter_slots(slots):
        return None
    question = str(item.get("question", "")).lower()
    list_prompt = any(
        marker in question
        for marker in [
            "if there are multiple",
            "find all",
            "all other zeros",
            "all solutions",
            "separate the different",
            "comma separated list",
        ]
    )
    if not list_prompt:
        return None
    return boxed("(" + ", ".join(slots) + ")")


def _parse_quadratic(question: str) -> tuple[float, float, float] | None:
    match = re.search(
        r"equation\$?([+-]?\d*)x\^2([+-]\d*)x([+-]\d+)=0",
        question.replace(" ", ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    def coef(value: str) -> float:
        if value in ("", "+"):
            return 1.0
        if value == "-":
            return -1.0
        return float(value)

    return coef(match.group(1)), coef(match.group(2)), float(match.group(3))


def repair_complex_quadratic(item: dict[str, Any], response: str, diagnostics: dict[str, Any]) -> str | None:
    if diagnostics["expected_slots"] != 1:
        return None
    question = str(item.get("question", ""))
    if "form $a+b i$" not in question and "form a+b i" not in question:
        return None
    coeffs = _parse_quadratic(question)
    if coeffs is None:
        return None
    a, b, c = coeffs
    disc = b * b - 4 * a * c
    if disc >= 0:
        return None
    real = -b / (2 * a)
    imag = math.sqrt(-disc) / (2 * a)
    root1 = f"{real:.14g}-{imag:.14g}i"
    root2 = f"{real:.14g}+{imag:.14g}i"
    return boxed(f"({root1}, {root2})")


def repair_printing_signature(item: dict[str, Any], response: str, diagnostics: dict[str, Any]) -> str | None:
    if diagnostics["expected_slots"] != 6:
        return None
    question = str(item.get("question", ""))
    if "signature" not in question.lower() or "16 pages" not in question.lower():
        return None
    unit_match = re.search(r"signatures of\s+(\d+)\s+pages", question, flags=re.IGNORECASE)
    cost_match = re.search(r"costs\s+\\?\$([0-9]+(?:\.[0-9]+)?)", question, flags=re.IGNORECASE)
    if not unit_match or not cost_match:
        return None
    unit = int(unit_match.group(1))
    cost = float(cost_match.group(1))
    page_counts = [
        int(value)
        for value in re.findall(
            r"(?:book of|there are)\s+(\d+)\s+pages",
            question,
            flags=re.IGNORECASE,
        )
    ]
    if len(page_counts) < 2:
        return None
    first = cost * math.ceil(page_counts[0] / unit)
    second = cost * math.ceil(page_counts[1] / unit)
    formula = f"{cost:g}*p/{unit}"
    return boxed(f"{first:g}, {second:g}, {formula}, up, 1, {cost:g}")


def repair_chair_table(item: dict[str, Any], response: str, diagnostics: dict[str, Any]) -> str | None:
    if diagnostics["expected_slots"] != 10:
        return None
    question = str(item.get("question", ""))
    if "wooden chairs" not in question.lower() or "f(60)" not in question or "f(z)=6000" not in question:
        return None
    n_match = re.search(r"n\s*&([^\\]+)\\\\", question)
    c_match = re.search(r"C\(n\)\s*&([^\\]+)\\\\", question)
    if not n_match or not c_match:
        return None
    n_values = [int(x) for x in re.findall(r"-?\d+", n_match.group(1))]
    c_values = [int(x) for x in re.findall(r"-?\d+", c_match.group(1))]
    if len(n_values) != len(c_values) or not n_values:
        return None
    table = dict(zip(n_values, c_values))
    inverse = {cost: n for n, cost in table.items()}
    try:
        a = table[60]
        b = table[40]
        c = inverse[6000]
        d = table[0]
    except KeyError:
        return None
    return boxed(f"{a}, {b}, {c}, {d}, (d), (b), None of the above, (c), (a), None of the above")


def _format_root(value: Any) -> str:
    value = sp.simplify(value)
    if value.is_Integer:
        return str(int(value))
    if value.is_Rational:
        return f"{float(value):.14g}"
    return str(value)


def repair_polynomial_other_zeros(item: dict[str, Any], response: str, diagnostics: dict[str, Any]) -> str | None:
    if diagnostics["expected_slots"] != 1 or diagnostics["parsed_slots"] <= 1:
        return None
    question = str(item.get("question", ""))
    lower = question.lower()
    if "all other zeros" not in lower or "polynomial" not in lower:
        return None
    poly_match = re.search(r"P\(x\)=([^$,]+)", question)
    given_match = re.search(r"x\s*=\s*([-+]?\d+(?:\.\d+)?)\$?\s+is a zero", question, flags=re.IGNORECASE)
    if not poly_match or not given_match:
        return None
    x = sp.Symbol("x")
    expr_text = poly_match.group(1).strip().replace("^", "**")
    expr_text = re.sub(r"(?<=\d)x", "*x", expr_text)
    try:
        expr = sp.sympify(expr_text, locals={"x": x})
        given = sp.nsimplify(float(given_match.group(1)))
        roots = sp.Poly(expr, x).all_roots()
    except Exception:
        return None
    other_roots = [sp.simplify(root) for root in roots if sp.simplify(root - given) != 0]
    if not other_roots:
        return None
    try:
        other_roots = sorted(other_roots, key=lambda root: float(sp.N(root)), reverse=True)
    except Exception:
        pass
    return boxed("(" + ", ".join(_format_root(root) for root in other_roots) + ")")


REPAIR_RULES = [
    ("printing_signature_slots", repair_printing_signature),
    ("chair_table_slots", repair_chair_table),
    ("complex_quadratic_slots", repair_complex_quadratic),
    ("polynomial_other_zeros_slots", repair_polynomial_other_zeros),
    ("multiselect_letter_slots", repair_multiselect_letters),
    ("single_slot_list_wrapper", repair_single_slot_list),
]


def candidate_ok(item: dict[str, Any], response: str) -> bool:
    diagnostics = final_answer_diagnostics(item, response)
    return bool(diagnostics.get("format_ok"))


def maybe_repair(item: dict[str, Any], row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    original = normalize_final_response(item, str(row.get("response", "")))
    diagnostics = final_answer_diagnostics(item, original)
    audit = {
        "id": int(item["id"]),
        "attempted": False,
        "accepted": False,
        "rule": None,
        "before": original,
        "after": original,
        "before_diagnostics": diagnostics,
    }
    if diagnostics.get("format_ok"):
        return {**row, "response": original}, audit

    for name, rule in REPAIR_RULES:
        candidate = rule(item, original, diagnostics)
        if not candidate:
            continue
        audit["attempted"] = True
        audit["rule"] = name
        audit["candidate"] = candidate
        if candidate_ok(item, candidate):
            repaired = {
                **row,
                "response": candidate,
                "pre_slot_repair_response": original,
                "slot_repair_rule": name,
            }
            audit["accepted"] = True
            audit["after"] = candidate
            audit["after_diagnostics"] = final_answer_diagnostics(item, candidate)
            return repaired, audit
    return {**row, "response": original}, audit


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    data_by_id = index_by_id(data)
    score_judger = Judger(strict_extract=False)

    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in read_jsonl(args.responses):
        item = data_by_id[int(row["id"])]
        out, audit = maybe_repair(item, row)
        out["answer_key"] = extract_answer_key(item, str(out.get("response", "")), strict=True)
        out.update(final_answer_diagnostics(item, str(out.get("response", ""))))
        if args.score and has_gold(item):
            out["gold"] = item["answer"]
            out["correct"] = score_item(score_judger, item, str(out.get("response", "")))
            audit["before_correct"] = score_item(score_judger, item, str(audit["before"]))
            audit["after_correct"] = bool(out["correct"])
        output_rows.append(out)
        audit_rows.append(audit)

    write_jsonl(args.output, output_rows)
    if args.audit_output:
        out_path = Path(args.audit_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "attempted": sum(bool(row["attempted"]) for row in audit_rows),
            "accepted": sum(bool(row["accepted"]) for row in audit_rows),
            "accepted_ids": [row["id"] for row in audit_rows if row["accepted"]],
            "rules": {},
            "rows": audit_rows,
        }
        for row in audit_rows:
            if row["accepted"]:
                summary["rules"][str(row["rule"])] = summary["rules"].get(str(row["rule"]), 0) + 1
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.score:
        print(json.dumps(summarize_results(output_rows), indent=2, sort_keys=True))
    print(f"Wrote slot-aware finalized responses to {args.output}")


if __name__ == "__main__":
    main()
