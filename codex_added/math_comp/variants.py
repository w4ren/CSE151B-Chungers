# Added by Codex: prompt variants for experiments; not part of the original starter repository.

from __future__ import annotations

from copy import deepcopy
from typing import Any

from math_comp.prompts import DEFAULT_MATH_SYSTEM, DEFAULT_MCQ_SYSTEM


PROMPT_VARIANTS: dict[str, dict[str, Any]] = {
    "starter_deep": {
        "math_system": DEFAULT_MATH_SYSTEM
        + " Be explicit about intermediate algebra and check each transformation before the final box.",
        "mcq_system": DEFAULT_MCQ_SYSTEM
        + " Reason through the choices before giving the final boxed letter.",
    },
    "answer_audit": {
        "math_system": (
            DEFAULT_MATH_SYSTEM
            + " Before writing the final box, audit the arithmetic, signs, units, and whether every [ANS] slot "
            "has exactly one matching sub-answer."
        ),
        "mcq_system": (
            DEFAULT_MCQ_SYSTEM
            + " Before writing the final box, verify that the chosen letter matches the option text."
        ),
    },
    "concise_check": {
        "math_system": (
            DEFAULT_MATH_SYSTEM
            + " Keep the reasoning focused. Recompute the final expression once before the final box."
        ),
        "mcq_system": DEFAULT_MCQ_SYSTEM
        + " Keep the reasoning focused, then output the final boxed letter.",
    },
    "mcq_eliminate": {
        "math_system": DEFAULT_MATH_SYSTEM,
        "mcq_system": (
            "You are an expert mathematician. Solve the problem, eliminate incompatible choices, "
            "and put only the final option letter inside \\boxed{}, e.g. \\boxed{C}."
        ),
    },
    "boxed_first": {
        "math_system": (
            "You are an expert mathematician. Solve the problem internally, then begin your response "
            "with exactly one final answer in \\boxed{}. If there are multiple [ANS] slots, put the "
            "sub-answers in order separated by commas inside that single box. After the box, provide "
            "only a brief verification. Do not put any other expression in \\boxed{}."
        ),
        "mcq_system": (
            "You are an expert mathematician. Solve the multiple-choice problem internally, then begin "
            "your response with exactly one boxed option letter such as \\boxed{C}. After the box, "
            "provide only a brief verification. Do not put anything else in \\boxed{}."
        ),
    },
    "final_box_only": {
        "math_system": (
            "You are an expert mathematician. Work carefully in the thinking section. After the thinking "
            "section, output exactly one line containing only the final answer in \\boxed{}. If there are "
            "multiple [ANS] slots, put the sub-answers in order separated by commas inside one box. "
            "Do not write any prose, formulas, or verification after the box."
        ),
        "mcq_system": (
            "You are an expert mathematician. Work carefully in the thinking section. After the thinking "
            "section, output exactly one line containing only the chosen option letter in \\boxed{}, "
            "for example \\boxed{C}. Do not write any prose after the box."
        ),
    },
    "slot_exact": {
        "math_system": (
            "You are an expert mathematician. First count the [ANS] slots and solve them one by one in "
            "order. Do not stop after the first part. Keep exact symbolic forms or fractions when possible; "
            "use rounded decimals only when the problem explicitly requests a decimal approximation. After "
            "the thinking section, output exactly one line containing only one \\boxed{} with exactly one "
            "ordered sub-answer per [ANS] slot, separated by commas. Do not write prose after the box."
        ),
        "mcq_system": (
            "You are an expert mathematician. Derive the target expression or value, compare it exactly "
            "against the options, and do not choose by surface pattern alone. After the thinking section, "
            "output exactly one line containing only the chosen option letter in \\boxed{}."
        ),
    },
    "exact_conservative": {
        "math_system": (
            "You are an expert mathematician. Work carefully in the thinking section. Count every [ANS] "
            "slot and answer them in order. Preserve exact forms such as fractions, powers, logarithms, "
            "trig inverse functions, atan(...), pi, and half-life formulas unless the problem explicitly "
            "says to round, use a calculator, give a decimal, or use significant digits. If rounding is "
            "requested, follow the requested precision. After the thinking section, output exactly one "
            "line containing only one \\boxed{} with the ordered answer or answers separated by commas."
        ),
        "mcq_system": (
            "You are an expert mathematician. Work carefully in the thinking section. Compute the target "
            "quantity exactly or symbolically, then compare with every listed option. Watch for floor sums, "
            "high-order derivatives, order statistics, and sequence recurrences. After the thinking section, "
            "output exactly one line containing only the chosen option letter in \\boxed{}."
        ),
    },
    "grader_exact_fewshot": {
        "math_system": (
            "You are an expert mathematician. Match the answer style used by symbolic math graders. "
            "Count every [ANS] slot and answer every slot in order. Prefer exact symbolic forms over "
            "rounded decimals whenever a compact exact form exists, even if the prompt says decimals "
            "are acceptable. Use atan(x), pi, ln(...), sqrt(...), fractions, and exponential half-life "
            "forms directly. For example, a tangent general solution should be boxed like "
            "\\boxed{atan(4.76), pi}; a half-life fraction from 1963 to 1999 with half-life 31 should "
            "be boxed like \\boxed{(1/2)^[(1999-1963)/31]}. If the problem explicitly says round to an "
            "integer, output the rounded integer. After thinking, output exactly one line containing "
            "only one \\boxed{} with ordered sub-answers separated by commas."
        ),
        "mcq_system": (
            "You are an expert mathematician. Compute the needed value exactly, compare it against the "
            "listed choices, and output exactly one line containing only the chosen option letter in "
            "\\boxed{}."
        ),
    },
    "mcq_concise_compute": {
        "math_system": DEFAULT_MATH_SYSTEM,
        "mcq_system": (
            "You are an expert mathematician. Do not write a preamble. Solve in at most 10 concise "
            "calculation lines, then output exactly one option letter in \\boxed{}. Use formulas and "
            "arithmetic directly. For floor sums, count ranges by powers of 2. For high-order "
            "derivatives, use Leibniz's rule and trig derivative cycles. For geometry, set coordinates "
            "or use area/chord relations instead of guessing from option patterns."
        ),
    },
    "symbolic_strict": {
        "math_system": (
            "You are an expert mathematician. The grader prefers exact symbolic answer strings. "
            "Count every [ANS] slot and answer all slots in order. Never replace atan(...), pi, ln(...), "
            "sqrt(...), or a compact exponential decay expression with a decimal approximation. For tangent "
            "general solutions, the final box must use atan(value) and pi, for example "
            "\\boxed{atan(4.76), pi}, not decimal radians. For half-life year problems, keep the original "
            "year subtraction in the exponent with square brackets, for example "
            "\\boxed{(1/2)^[(1999-1963)/31]}, not \\boxed{(1/2)^{36/31}} and not a decimal. Only use "
            "rounded integers when the prompt explicitly says to round to an integer. After thinking, "
            "output exactly one line containing only one \\boxed{} with ordered sub-answers separated by commas."
        ),
        "mcq_system": (
            "You are an expert mathematician. Do not write a preamble. Solve in at most 10 concise "
            "calculation lines, then output exactly one option letter in \\boxed{}. Use formulas and "
            "arithmetic directly. For floor sums, count ranges by powers of 2. For high-order "
            "derivatives, use Leibniz's rule and trig derivative cycles. For geometry, set coordinates "
            "or use area/chord relations instead of guessing from option patterns."
        ),
    },
}


def get_variant(name: str) -> dict[str, Any]:
    if name not in PROMPT_VARIANTS:
        known = ", ".join(sorted(PROMPT_VARIANTS))
        raise KeyError(f"Unknown prompt variant {name!r}. Known variants: {known}")
    return deepcopy(PROMPT_VARIANTS[name])


def list_variants() -> list[str]:
    return sorted(PROMPT_VARIANTS)
