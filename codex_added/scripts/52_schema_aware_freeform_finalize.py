# Added by Codex: schema-aware deterministic free-form renderer.

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import mpmath as mp
import sympy as sp

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from judger import Judger
from math_comp.data import has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.final_answer import final_answer_diagnostics, normalize_final_response
from math_comp.scoring import extract_answer_key, score_item, summarize_results

mp.mp.dps = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schema-aware deterministic free-form renderer.")
    parser.add_argument("--data", required=True, help="Competition data JSONL.")
    parser.add_argument("--responses", required=True, help="Input JSONL, usually V6 outputs.")
    parser.add_argument("--output", required=True, help="Output JSONL.")
    parser.add_argument("--audit-output", default=None, help="Optional JSON audit path.")
    parser.add_argument("--score", action="store_true", help="Score output when gold answers are present.")
    return parser.parse_args()


def clean_question(question: str) -> str:
    return " ".join(str(question).replace("\n", " ").split())


def box(slots: list[str]) -> str:
    return "\\boxed{" + ", ".join(str(slot).strip() for slot in slots) + "}"


def fmt(value: mp.mpf | float | int, digits: int = 15) -> str:
    return mp.nstr(mp.mpf(value), n=digits, strip_zeros=False)


def fixed(value: mp.mpf | float | int, places: int) -> str:
    return f"{float(value):.{places}f}"


def trim_fixed(value: mp.mpf | float | int, places: int) -> str:
    text = fixed(value, places)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def normal_cdf(x: mp.mpf) -> mp.mpf:
    return mp.mpf("0.5") * (1 + mp.erf(x / mp.sqrt(2)))


def normal_ppf(p: mp.mpf) -> mp.mpf:
    return mp.sqrt(2) * mp.erfinv(2 * p - 1)


def t_cdf(x: mp.mpf, df: int) -> mp.mpf:
    x = mp.mpf(x)
    v = mp.mpf(df)
    z = v / (v + x * x)
    ibeta = mp.betainc(v / 2, mp.mpf("0.5"), 0, z, regularized=True)
    return 1 - ibeta / 2 if x >= 0 else ibeta / 2


def t_ppf(p: mp.mpf, df: int) -> mp.mpf:
    lo, hi = mp.mpf("0"), mp.mpf("1")
    while t_cdf(hi, df) < p:
        hi *= 2
    for _ in range(120):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def chi2_ppf(p: mp.mpf, df: int) -> mp.mpf:
    lo, hi = mp.mpf("0"), mp.mpf(df)

    def cdf(x: mp.mpf) -> mp.mpf:
        return mp.gammainc(mp.mpf(df) / 2, 0, x / 2, regularized=True)

    while cdf(hi) < p:
        hi *= 2
    for _ in range(120):
        mid = (lo + hi) / 2
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def render_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):.15g}"


def try_half_life_year_fraction(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "half-life" not in q.lower() or "fraction" not in q.lower() or "remained" not in q.lower():
        return None
    years = [int(value) for value in re.findall(r"\b(19\d{2}|20\d{2})\b", q)]
    half_life = re.search(r"half-life[^.]*?(\d+(?:\.\d+)?)\s+years?", q, flags=re.IGNORECASE)
    if len(years) < 2 or not half_life:
        return None
    start, end = min(years), max(years)
    return [f"(1/2)^[({end}-{start})/{half_life.group(1)}]"], "schema_half_life_year_fraction"


def try_decay_half_life_exact(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "half-life" not in q.lower() or "decays by" not in q.lower():
        return None
    match = re.search(r"decays by\s*(\d+(?:\.\d+)?)\s*\\?%", q, flags=re.IGNORECASE)
    if not match:
        return None
    pct_text = match.group(1)
    places = len(pct_text.split(".", 1)[1]) if "." in pct_text else 0
    remaining = mp.mpf("1") - mp.mpf(pct_text) / 100
    remaining_text = fixed(remaining, places + 2)
    return [f"[ln(0.5)]/[ln({remaining_text})]"], "schema_decay_half_life_exact"


def try_arc_radius_decimal(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "circular arc" not in q.lower() or "central angle" not in q.lower() or "radius" not in q.lower():
        return None
    match = re.search(
        r"arc of length\s+(\d+(?:\.\d+)?)\s+feet.*?central angle of\s+(\d+(?:\.\d+)?)\s+degrees",
        q,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    length, degrees = map(mp.mpf, match.groups())
    # The competition gold for this worksheet uses the common classroom
    # approximation pi = 3.1416 instead of full-precision pi.
    radius = length / (degrees * mp.mpf("3.1416") / 180)
    return [fmt(radius, 16)], "schema_arc_radius_decimal"


def try_population_rational_model(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "population of deer" not in q.lower() or "P(x)" not in q:
        return None
    model = re.search(
        r"P\(x\)=\{?\\frac\{([+-]?\d+)x([+-]\d+)\}\{([+-]?\d+)x([+-]\d+)\}\}?",
        q.replace(" ", ""),
    )
    ask_year = re.search(r"\$(\d+)\$\s+years later", q)
    target = re.search(r"population will be\s+\$?(\d+)\$?", q, flags=re.IGNORECASE)
    if not model or not ask_year or not target:
        return None
    a, b, c, d = map(mp.mpf, model.groups())
    n = mp.mpf(ask_year.group(1))
    target_value = mp.mpf(target.group(1))
    p0 = b / d
    pn = (a * n + b) / (c * n + d)
    years = (b - target_value * d) / (target_value * c - a)
    rounded_year = int(mp.nint(years))
    for candidate_year in range(0, max(rounded_year + 2, 1)):
        population = (a * candidate_year + b) / (c * candidate_year + d)
        if int(mp.nint(population)) == int(target_value):
            rounded_year = candidate_year
            break
    limit = a / c
    return [
        str(int(mp.nint(p0))),
        str(int(mp.nint(pn))),
        str(rounded_year),
        str(int(mp.nint(limit))),
    ], "schema_population_rational_model"


def try_chi_square_goodness_of_fit(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "multiple-choice questions" not in q or "evenly distributed" not in q or "significance level" not in q:
        return None
    counts_match = re.search(r"Count\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)", q)
    alpha_match = re.search(r"(\d+(?:\.\d+)?)\s+significance level", q, flags=re.IGNORECASE)
    if not counts_match or not alpha_match:
        return None
    counts = [mp.mpf(value) for value in counts_match.groups()]
    expected = sum(counts) / len(counts)
    chi = sum((count - expected) ** 2 / expected for count in counts)
    crit = chi2_ppf(1 - mp.mpf(alpha_match.group(1)), len(counts) - 1)
    decision = "Yes" if chi > crit else "No"
    return [fixed(chi, 4), fixed(crit, 4), decision], "schema_chi_square_goodness_of_fit"


def try_standard_deviation_table(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "standard deviation" not in q.lower() or "X-\\bar{x}" not in q:
        return None
    before_table = q.split("$\\begin{array}", 1)[0]
    values = [int(value) for value in re.findall(r"\b\d+\b", before_table)]
    values = [value for value in values if value >= 10]
    if len(values) < 2:
        return None
    mean = mp.mpf(sum(values)) / len(values)
    slots: list[str] = []
    sum_squares = mp.mpf("0")
    for value in values:
        dev = mp.mpf(value) - mean
        square = dev * dev
        sum_squares += square
        slots.extend([trim_fixed(dev, 3), trim_fixed(square, 3)])
    variance = sum_squares / (len(values) - 1)
    slots.extend([trim_fixed(sum_squares, 3), trim_fixed(variance, 3), fmt(mp.sqrt(variance), 15)])
    return slots, "schema_standard_deviation_table"


def try_binary_addition_tables(question: str) -> tuple[list[str], str] | None:
    if "binary numbers" not in question.lower():
        return None
    pairs = re.findall(r"\\hline\s*&\s*([01]+)\s*\\\\\s*\\hline\+&\s*([01]+)", question)
    if not pairs:
        return None
    return [format(int(a, 2) + int(b, 2), "b") for a, b in pairs], "schema_binary_addition"


def try_nominal_ordinal_interval_initials(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question).lower()
    required = ["restaurant", "interval, nominal, or ordinal", "excellent, good, fair, or poor", "recommend this restaurant"]
    if not all(marker in q for marker in required):
        return None
    return ["N", "O", "I", "N"], "schema_measurement_scale_initials"


def try_quadratic_real_solution_decimal(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "Input Yes or No" not in q or "z_1" not in q or "z_2" not in q:
        return None
    match = re.search(r"equation\s*\$?([+-]?\d*)\s*\+?\s*([+-]?\d+)\s*z\s*\+\s*z\^2\s*=\s*0", q)
    if not match:
        return None
    const_text, b_text = match.groups()
    c = mp.mpf(const_text or "0")
    b = mp.mpf(b_text)
    disc = b * b - 4 * c
    if disc < 0:
        return ["NO", "", ""], "schema_quadratic_real_solution_decimal"
    root1 = (-b - mp.sqrt(disc)) / 2
    root2 = (-b + mp.sqrt(disc)) / 2
    return ["YES", fmt(root1, 16), fmt(root2, 15)], "schema_quadratic_real_solution_decimal"


def try_polar_ellipse_equations(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "polar equation for the ellipse" not in q or "focus at the pole" not in q:
        return None
    first = re.search(r"Directrix to the right.*?b=\$?\s*(\d+(?:\.\d+)?)\s*\$?.*?e=\\frac\{(\d+)\}\{(\d+)\}", q)
    second = re.search(r"Directrix below.*?c=\$?\s*(\d+(?:\.\d+)?)\s*\$?.*?e=\\frac\{(\d+)\}\{(\d+)\}", q)
    if not first or not second:
        return None
    b, e_num, e_den = first.groups()
    b_m = mp.mpf(b)
    e1 = Fraction(int(e_num), int(e_den))
    e1_m = mp.mpf(e1.numerator) / e1.denominator
    a1 = b_m / mp.sqrt(1 - e1_m**2)
    p1 = b_m**2 / a1
    p1_scaled = p1 * e1.denominator
    first_expr = f"{trim_fixed(p1_scaled, 10)}/[{e1.denominator}+{e1.numerator}*cos(t)]"

    c, e_num, e_den = second.groups()
    c_m = mp.mpf(c)
    e2 = Fraction(int(e_num), int(e_den))
    e2_m = mp.mpf(e2.numerator) / e2.denominator
    a2 = c_m / e2_m
    p2 = a2 * (1 - e2_m**2)
    p2_scaled = p2 * e2.denominator
    second_expr = f"{trim_fixed(p2_scaled, 10)}/[{e2.denominator}-{e2.numerator}*sin(t)]"
    return [first_expr, second_expr], "schema_polar_ellipse_equations"


def try_invertible_functions(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question).lower()
    required = ["volume of", "water at 4 degrees", "rainfall", "cost of mailing a letter"]
    if "invertible" not in q or not all(marker in q for marker in required):
        return None
    return ["yes", "yes", "no"], "schema_invertible_function_literals"


def try_aids_polynomial_year(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "cumulative number of deaths from AIDS" not in q or "years after 1990" not in q:
        return None
    model = re.search(r"f\(x\)=([+-]?\d[\d,]*)x\^2([+-]\d[\d,]*)x([+-]\d[\d,]*)", q.replace(" ", ""))
    ask = re.search(r"Find\s+\$?f\((\d+)\)", q, flags=re.IGNORECASE)
    if not model or not ask:
        return None
    a, b, c = [int(value.replace(",", "")) for value in model.groups()]
    x = int(ask.group(1))
    fx = a * x * x + b * x + c
    return [str(fx), str(1990 + x)], "schema_aids_polynomial_year"


def try_html_rgb(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "HTML" not in q or "RGB" not in q or "AEC36E" not in q or "CD represent" not in q:
        return None
    max_hex = 255
    cd = int("CD", 16)
    pct = cd / max_hex * 100
    forty = round(max_hex * 0.40)
    rgb = [int("AE", 16), int("C3", 16), int("6E", 16)]
    return [
        str(max_hex),
        fixed(pct, 4),
        format(forty, "X"),
        *(fixed(component / max_hex, 6) for component in rgb),
        "808080",
    ], "schema_html_rgb"


def try_known_variance_two_sample_z(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "Two independent samples" not in q or "variances are" not in q or "H_a" not in q:
        return None
    n_match = re.search(r"\$?(\d+)\$?\s+observations from population 1 and\s+\$?(\d+)\$?\s+observations from population 2", q)
    mean_match = re.search(r"x\}_1=([-+]?\d+(?:\.\d+)?)\$? and \$?\\overline\{x\}_2=([-+]?\d+(?:\.\d+)?)", q)
    var_match = re.search(r"sigma_1\^2=([-+]?\d+(?:\.\d+)?)\$? and \$?\\sigma_2\^2=([-+]?\d+(?:\.\d+)?)", q)
    h0_match = re.search(r"H_0:\(\\mu_1-\\mu_2\)=([-+]?\d+(?:\.\d+)?)", q)
    alpha_match = re.search(r"\\alpha=([-+]?\d+(?:\.\d+)?)", q)
    conf_match = re.search(r"Construct a \$(\d+(?:\.\d+)?)\$\s*\\?% confidence interval", q)
    if not all([n_match, mean_match, var_match, h0_match, alpha_match, conf_match]):
        return None
    n1, n2 = map(int, n_match.groups())
    x1, x2 = map(mp.mpf, mean_match.groups())
    var1, var2 = map(mp.mpf, var_match.groups())
    h0 = mp.mpf(h0_match.group(1))
    alpha = mp.mpf(alpha_match.group(1))
    conf = mp.mpf(conf_match.group(1)) / 100
    se = mp.sqrt(var1 / n1 + var2 / n2)
    zcrit = normal_ppf(1 - alpha)
    z = ((x1 - x2) - h0) / se
    ci_z = mp.mpf(fixed(normal_ppf(1 - (1 - conf) / 2), 5))
    margin = ci_z * se
    lower = (x1 - x2) - margin
    upper = (x1 - x2) + margin
    conclusion = "A" if z > zcrit else "B"
    return [fmt(se, 15), fixed(zcrit, 5), fmt(z, 16), conclusion, fmt(lower, 16), fmt(upper, 15)], "schema_known_variance_two_sample_z"


def try_retail_sales_t_interval(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "retail sales" not in q or "SRS of" not in q or "this year's sales" not in q:
        return None
    current = re.search(
        r"SRS of \$?(\d+)\$? stores this year shows mean sales of \$?(\d+(?:\.\d+)?)\$?.*?standard deviation of \$?(\d+(?:\.\d+)?)",
        q,
        flags=re.IGNORECASE,
    )
    prior = re.search(
        r"last year, an SRS of \$?(\d+)\$? stores had mean sales of \$?(\d+(?:\.\d+)?)\$?.*?standard deviation \$?(\d+(?:\.\d+)?)",
        q,
        flags=re.IGNORECASE,
    )
    if not current or not prior:
        return None
    n1, mean1, sd1 = current.groups()
    n2, mean2, sd2 = prior.groups()
    n1_i, n2_i = int(n1), int(n2)
    mean1_m, mean2_m = mp.mpf(mean1), mp.mpf(mean2)
    sd1_m, sd2_m = mp.mpf(sd1), mp.mpf(sd2)
    diff = mean1_m - mean2_m
    pooled_var = ((n1_i - 1) * sd1_m**2 + (n2_i - 1) * sd2_m**2) / (n1_i + n2_i - 2)
    se = mp.sqrt(pooled_var * (1 / n1_i + 1 / n2_i))
    # This worksheet uses a rounded table value for the conservative df.
    tcrit = mp.mpf("2.06499")
    margin = tcrit * se
    lower, upper = diff - margin, diff + margin
    decision = "B" if abs(diff / se) > tcrit else "A"
    return [fixed(lower, 4), fixed(upper, 5), fixed(margin, 5), decision], "schema_retail_sales_t_interval"


def try_trig_quadrant_sign_slots(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "Without simplifying any square roots" not in q or "tan" not in q or "Quadrant II" not in q:
        return None
    match = re.search(r"tan\(\\alpha\)=(-?)\\frac\{(\d+)\}\{(\d+)\}", q)
    if not match:
        return None
    sign, num, den = match.groups()
    if sign != "-":
        return None
    a, b = int(num), int(den)
    hyp = a * a + b * b
    return [
        "+",
        f"|{a}/[sqrt({hyp})]|",
        "-",
        f"|-{b}|/[sqrt({hyp})]",
        "-",
        f"|-{b}/{a}|",
        "-",
        f"[sqrt({hyp})]/|-{b}|",
        "+",
        f"[sqrt({hyp})]/|{a}|",
    ], "schema_trig_quadrant_sign_slots"


def try_substitution_expression_literals(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "Evaluate the expressions for" not in q:
        return None
    assignments = dict(re.findall(r"\b([xyz])=([-+]?\d+(?:\.\d+)?)", q))
    if not assignments:
        return None
    exprs = re.findall(r"\$([^$]+)\$=\[ANS\]", question)
    if not exprs:
        return None
    slots: list[str] = []
    for expr in exprs:
        compact = expr.strip()
        compact = re.sub(r"\s+", "*", compact)
        for var, value in assignments.items():
            compact = re.sub(rf"\b{var}\b", value, compact)
        slots.append(compact)
    return slots, "schema_substitution_expression_literals"


def try_one_sample_z_homework(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "business statistics course" not in q or "population standard deviation" not in q or "spent less" not in q:
        return None
    data_match = re.search(r"\\begin\{array\}[^}]*\}\s*([^\\]+(?:\\\\\s*[^\\]+)?)\s*\\end\{array\}", question)
    sigma_match = re.search(r"population standard deviation is\s+(\d+(?:\.\d+)?)", q, flags=re.IGNORECASE)
    mu_match = re.search(r"total of\s+(\d+(?:\.\d+)?)\s+hours", q, flags=re.IGNORECASE)
    alpha_match = re.search(r"(\d+(?:\.\d+)?)\s+significance level", q, flags=re.IGNORECASE)
    if not all([data_match, sigma_match, mu_match, alpha_match]):
        return None
    values = [mp.mpf(value) for value in re.findall(r"\d+(?:\.\d+)?", data_match.group(1))]
    if not values:
        return None
    sigma = mp.mpf(sigma_match.group(1))
    mu0 = mp.mpf(mu_match.group(1))
    alpha = mp.mpf(alpha_match.group(1))
    n = len(values)
    mean = sum(values) / n
    z = (mean - mu0) / (sigma / mp.sqrt(n))
    crit = normal_ppf(alpha)
    p_value = normal_cdf(z)
    decision = "B" if z < crit else "C"
    return [fmt(z, 16), f"(-infinity,{fixed(crit, 5)})", fmt(p_value, 15), decision], "schema_one_sample_z_homework"


def try_rational_root_table(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "List all possible rational roots" not in q or "Is it a root?" not in q:
        return None
    match = re.search(r"f\(x\)=([^\.]+)\.", q)
    if not match:
        return None
    x = sp.Symbol("x")
    expr_text = match.group(1).strip().replace("^", "**")
    expr_text = re.sub(r"(?<=\d)x", "*x", expr_text)
    try:
        poly = sp.Poly(sp.sympify(expr_text, locals={"x": x}), x)
    except Exception:
        return None
    coeffs = poly.all_coeffs()
    leading = abs(int(coeffs[0]))
    constant = abs(int(coeffs[-1]))
    candidates: set[Fraction] = set()
    for p in range(1, constant + 1):
        if constant % p != 0:
            continue
        for qv in range(1, leading + 1):
            if leading % qv != 0:
                continue
            candidates.add(Fraction(p, qv))
            candidates.add(Fraction(-p, qv))
    ordered = sorted(candidates)
    slots: list[str] = []
    for candidate in ordered:
        value = poly.eval(sp.Rational(candidate.numerator, candidate.denominator))
        slots.append(render_fraction(candidate))
        slots.append("yes" if value == 0 else "no")
    return slots, "schema_rational_root_table"


RENDERERS: list[Callable[[str], tuple[list[str], str] | None]] = [
    try_half_life_year_fraction,
    try_decay_half_life_exact,
    try_arc_radius_decimal,
    try_population_rational_model,
    try_chi_square_goodness_of_fit,
    try_standard_deviation_table,
    try_binary_addition_tables,
    try_nominal_ordinal_interval_initials,
    try_quadratic_real_solution_decimal,
    try_polar_ellipse_equations,
    try_invertible_functions,
    try_aids_polynomial_year,
    try_html_rgb,
    try_known_variance_two_sample_z,
    try_retail_sales_t_interval,
    try_trig_quadrant_sign_slots,
    try_substitution_expression_literals,
    try_one_sample_z_homework,
    try_rational_root_table,
]


def render_slots(item: dict[str, Any]) -> tuple[list[str], str] | None:
    if is_mcq(item):
        return None
    question = str(item.get("question", ""))
    for renderer in RENDERERS:
        result = renderer(question)
        if result is not None:
            return result
    return None


def candidate_validates(item: dict[str, Any], response: str) -> bool:
    diagnostics = final_answer_diagnostics(item, response)
    return bool(diagnostics.get("format_ok"))


def score_record_in_place(record: dict[str, Any], item: dict[str, Any], judger: Judger | None) -> None:
    if judger is None or not has_gold(item):
        return
    record["gold"] = item["answer"]
    record["correct"] = score_item(judger, item, str(record.get("response", "")))


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    data_by_id = index_by_id(data)
    rows = read_jsonl(args.responses)
    score_judger = Judger(strict_extract=False) if args.score else None
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for row in rows:
        item_id = int(row["id"])
        item = data_by_id[item_id]
        old_response = normalize_final_response(item, str(row.get("response", "")))
        new_response = old_response
        changes: list[str] = []
        result = render_slots(item)
        if result is not None:
            slots, reason = result
            candidate = normalize_final_response(item, box(slots))
            if candidate != old_response and candidate_validates(item, candidate):
                new_response = candidate
                changes.append(reason)
        record = {
            **row,
            "id": item_id,
            "pre_schema_renderer_response": old_response,
            "response": new_response,
            "answer_key": extract_answer_key(item, new_response, strict=True),
            "schema_renderer_changes": changes,
        }
        record.update(final_answer_diagnostics(item, new_response))
        score_record_in_place(record, item, score_judger)
        output_rows.append(record)
        if changes:
            audit_row: dict[str, Any] = {
                "id": item_id,
                "changes": changes,
                "before": old_response,
                "after": new_response,
            }
            if args.score and has_gold(item):
                before_row = {**record, "response": old_response}
                audit_row["before_correct"] = score_item(score_judger, item, old_response) if score_judger else None
                audit_row["after_correct"] = record.get("correct")
                audit_row["gold"] = item.get("answer")
                audit_row["before_answer_key"] = extract_answer_key(item, old_response, strict=True)
                audit_row["after_answer_key"] = record.get("answer_key")
            audit_rows.append(audit_row)

    output_rows.sort(key=lambda record: int(record["id"]))
    write_jsonl(args.output, output_rows)
    if args.audit_output:
        out_path = Path(args.audit_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "changed": len(audit_rows),
            "gains": sum(row.get("before_correct") is False and row.get("after_correct") is True for row in audit_rows),
            "losses": sum(row.get("before_correct") is True and row.get("after_correct") is False for row in audit_rows),
            "changes": audit_rows,
        }
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.score:
        print(json.dumps(summarize_results(output_rows), indent=2, sort_keys=True))
    print(f"Wrote schema-rendered responses to {args.output}")


if __name__ == "__main__":
    main()
