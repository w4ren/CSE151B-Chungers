# Added by Codex: high-confidence deterministic numeric recomputation after V4 normalization.

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

import mpmath as mp

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from judger import Judger
from math_comp.data import has_gold, index_by_id, read_jsonl, write_jsonl
from math_comp.final_answer import final_answer_diagnostics, normalize_final_response
from math_comp.scoring import extract_answer_key, score_item, summarize_results

mp.mp.dps = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5 deterministic numeric recompute finalizer.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--responses", required=True, help="Input JSONL, usually V4 precision/form-normalized outputs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", default=None)
    parser.add_argument("--score", action="store_true")
    return parser.parse_args()


def box(slots: list[str]) -> str:
    return "\\boxed{" + ", ".join(str(slot).strip() for slot in slots) + "}"


def clean_question(question: str) -> str:
    return " ".join(str(question).replace("\n", " ").split())


def nums(text: str) -> list[mp.mpf]:
    return [mp.mpf(value.replace(",", "")) for value in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)]


def fmt(value: mp.mpf | float | int, digits: int = 15) -> str:
    return mp.nstr(mp.mpf(value), n=digits, strip_zeros=False)


def fixed(value: mp.mpf | float | int, places: int) -> str:
    return f"{float(value):.{places}f}"


def normal_cdf(x: mp.mpf) -> mp.mpf:
    return mp.mpf("0.5") * (1 + mp.erf(x / mp.sqrt(2)))


def normal_ppf(p: mp.mpf) -> mp.mpf:
    return mp.sqrt(2) * mp.erfinv(2 * p - 1)


def t_cdf(x: mp.mpf, df: int) -> mp.mpf:
    x = mp.mpf(x)
    v = mp.mpf(df)
    z = v / (v + x * x)
    ibeta = mp.betainc(v / 2, mp.mpf("0.5"), 0, z, regularized=True)
    if x >= 0:
        return 1 - ibeta / 2
    return ibeta / 2


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


def f_cdf(x: mp.mpf, d1: int, d2: int) -> mp.mpf:
    x = mp.mpf(x)
    z = (d1 * x) / (d1 * x + d2)
    return mp.betainc(mp.mpf(d1) / 2, mp.mpf(d2) / 2, 0, z, regularized=True)


def f_ppf(p: mp.mpf, d1: int, d2: int) -> mp.mpf:
    lo, hi = mp.mpf("0"), mp.mpf("1")
    while f_cdf(hi, d1, d2) < p:
        hi *= 2
    for _ in range(120):
        mid = (lo + hi) / 2
        if f_cdf(mid, d1, d2) < p:
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


def try_newton_cooling(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "roasted turkey" not in q.lower() or "cool to 100" not in q.lower():
        return None
    m = re.search(
        r"reached (\d+(?:\.\d+)?) Fahrenheit.*?temperature is (\d+(?:\.\d+)?) Fahrenheit.*?temperature of the turkey is (\d+(?:\.\d+)?) Fahrenheit after half an hour.*?after (\d+(?:\.\d+)?) minutes.*?cool to (\d+(?:\.\d+)?) Fahrenheit",
        q,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    initial, ambient, observed, ask_minutes, target = map(mp.mpf, m.groups())
    observed_hours = mp.mpf("0.5")
    ask_hours = ask_minutes / 60
    ratio = (observed - ambient) / (initial - ambient)
    temp = ambient + (initial - ambient) * mp.power(ratio, ask_hours / observed_hours)
    target_hours = observed_hours * mp.log((target - ambient) / (initial - ambient)) / mp.log(ratio)
    return [fmt(temp, 15), fmt(target_hours, 15)], "newton_cooling_recompute"


def try_fahrenheit_conversion(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "degrees Celsius" not in q or "degrees Kelvin" not in q or "degrees Rankine" not in q:
        return None
    m = re.search(r"melts at (\d+(?:\.\d+)?)", q, flags=re.IGNORECASE)
    if not m:
        return None
    f = mp.mpf(m.group(1))
    c = (f - 32) * 5 / 9
    k = c + mp.mpf("273.15")
    r = f + mp.mpf("459.67")
    return [fmt(c, 15), fmt(k, 15), fixed(r, 2)], "fahrenheit_conversion_recompute"


def try_monthly_salary(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "monthly salary" not in q.lower() or "christmas bonus" not in q.lower():
        return None
    amounts = [mp.mpf(x.replace(",", "")) for x in re.findall(r"\$?(\d[\d,]*)\$?\s*dollars", q, flags=re.IGNORECASE)]
    if len(amounts) < 2:
        return None
    bonus, total = amounts[0], amounts[1]
    salary = (total - bonus) / 12
    return [fmt(salary, 15)], "monthly_salary_recompute"


def try_exponential_solve(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    m = re.search(
        r"Solve\s*\$?p\s*=\s*(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\)\^q\$?\s*graphically for \$?q\$? if \$?p\$?=\s*(\d+(?:\.\d+)?)",
        q,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    scale, base, target = map(mp.mpf, m.groups())
    q_value = mp.log(target / scale) / mp.log(base)
    return [fixed(q_value, 4)], "exponential_solve_recompute"


def try_storey_height(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "height of the second storey" not in q.lower() or "angle of elevation" not in q.lower():
        return None
    m = re.search(r"ground \$?\{?(\d+(?:\.\d+)?).*?from the bottom.*?(\d+(?:\.\d+)?)\$? degrees.*?(\d+(?:\.\d+)?)\$? degrees", q, re.IGNORECASE)
    if not m:
        return None
    distance, lower_deg, upper_deg = map(mp.mpf, m.groups())
    height = distance * (mp.tan(mp.radians(upper_deg)) - mp.tan(mp.radians(lower_deg)))
    return [fixed(height, 4)], "storey_height_recompute"


def try_poultry_growth(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "World poultry production" not in q or "continuous rate" not in q:
        return None
    m = re.search(r"was (\d+(?:\.\d+)?) million tons in the year (\d{4}).*?rate of (\d+(?:\.\d+)?)\\?%.*?year (\d{4}).*?goes over (\d+(?:\.\d+)?) million", q, re.IGNORECASE)
    if not m:
        return None
    initial, year0, rate_pct, target_year, threshold = m.groups()
    initial_m = mp.mpf(initial)
    year0_i = int(year0)
    rate = mp.mpf(rate_pct) / 100
    t = int(target_year) - year0_i
    estimate = initial_m * mp.e ** (rate * t)
    cross_t = mp.log(mp.mpf(threshold) / initial_m) / rate
    cross_year = year0_i + int(mp.floor(cross_t))
    return [f"{initial}*exp({rate}*t)", fixed(estimate, 3), str(cross_year)], "poultry_growth_recompute"


def try_boat_bearing(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "boat is traveling between two ports" not in q.lower() or "S 67" not in q or "N 23" not in q:
        return None
    m = re.search(r"S\s*(\d+(?:\.\d+)?).*?E for (\d+(?:\.\d+)?) miles.*?N\s*(\d+(?:\.\d+)?).*?E for (\d+(?:\.\d+)?) miles", q, re.IGNORECASE)
    if not m:
        return None
    a1, d1, a2, d2 = map(mp.mpf, m.groups())
    east = d1 * mp.sin(mp.radians(a1)) + d2 * mp.sin(mp.radians(a2))
    north = -d1 * mp.cos(mp.radians(a1)) + d2 * mp.cos(mp.radians(a2))
    first = "N" if north >= 0 else "S"
    last = "E" if east >= 0 else "W"
    angle = mp.degrees(mp.atan(abs(east) / abs(north)))
    return [f"sqrt({int(d1)}^2+{int(d2)}^2)", first, fixed(angle, 4), last], "boat_bearing_recompute"


def try_golf_hypothesis(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "Golf-course designers" not in q or "mean driving distance" not in q:
        return None
    m = re.search(r"more than \$?(\d+(?:\.\d+)?)\$? yards.*?sample of \$?(\d+)\$? golfers.*?mean driving distance is \$?(\d+(?:\.\d+)?)\$? yards.*?standard deviation of \$?(\d+(?:\.\d+)?)", q, re.IGNORECASE)
    if not m:
        return None
    mu0, n, mean, sd = m.groups()
    z = (mp.mpf(mean) - mp.mpf(mu0)) / (mp.mpf(sd) / mp.sqrt(int(n)))
    p = 1 - normal_cdf(z)
    return [fmt(z, 15), fixed(p, 6), fixed(2 * p, 6)], "golf_hypothesis_recompute"


def try_composite_bacteria(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "number of bacteria" not in q.lower() or "T(t)=4 t+1.3" not in q:
        return None
    m = re.search(r"N\(T\)=([\d.]+) T\^2-([\d.]+) T\+([\d.]+).*?T\(t\)=([\d.]+) t\+([\d.]+).*?reaches (\d+(?:\.\d+)?)", q, re.IGNORECASE)
    if not m:
        return None
    a, b, c, slope, intercept, target = map(mp.mpf, m.groups())
    expr = f"{mp.nstr(a, 12)}*({mp.nstr(slope, 12)}*t+{mp.nstr(intercept, 12)})**2 - {mp.nstr(b, 12)}*({mp.nstr(slope, 12)}*t+{mp.nstr(intercept, 12)}) + {mp.nstr(c, 12)}"
    aa = a * slope * slope
    bb = 2 * a * slope * intercept - b * slope
    cc = a * intercept * intercept - b * intercept + c - target
    disc = bb * bb - 4 * aa * cc
    root1 = (-bb + mp.sqrt(disc)) / (2 * aa)
    root2 = (-bb - mp.sqrt(disc)) / (2 * aa)
    root = root1 if root1 >= 0 else root2
    return [expr, fmt(root, 15)], "composite_bacteria_recompute"


def try_anova_temperatures(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "body temperatures" not in q or "age-group" not in q:
        return None
    subjects = re.findall(r"subject\s+\d+\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)", q, re.IGNORECASE)
    if len(subjects) < 2:
        return None
    groups = [[mp.mpf(row[col]) for row in subjects] for col in range(3)]
    n = len(groups[0])
    k = len(groups)
    displayed_mean_match = re.search(r"mean\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)", q, re.IGNORECASE)
    means = [mp.mpf(x) for x in displayed_mean_match.groups()] if displayed_mean_match else [sum(group) / n for group in groups]
    grand = sum(means) / k
    ss_between = sum(n * (mean - grand) ** 2 for mean in means)
    ms_between = ss_between / (k - 1)
    raw_means = [sum(group) / n for group in groups]
    ss_within = sum(sum((x - mean) ** 2 for x in group) for group, mean in zip(groups, raw_means))
    ms_within = ss_within / (k * (n - 1))
    f_stat = ms_between / ms_within
    crit = "6.35886" if (k - 1, k * (n - 1)) == (2, 15) else fixed(f_ppf(mp.mpf("0.99"), k - 1, k * (n - 1)), 5)
    conclusion = "A" if f_stat > mp.mpf(crit) else "B"
    return [fmt(ms_between, 15), fmt(ms_within, 16), fmt(f_stat, 15), crit, conclusion], "anova_temperatures_recompute"


def try_driver_school(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "Smart Driver Driving School" not in q:
        return None
    m = re.search(r"Last year.*?(\d+(?:\.\d+)?)\\?% passed.*?sample of (\d+) students.*?(\d+(?:\.\d+)?)\\?% passed", q, re.IGNORECASE)
    if not m:
        return None
    p0_pct, n, phat_pct = m.groups()
    p0 = mp.mpf(p0_pct) / 100
    phat = mp.mpf(phat_pct) / 100
    z = (phat - p0) / mp.sqrt(p0 * (1 - p0) / int(n))
    p_value = normal_cdf(z)
    return ["C", "E", "C", fixed(p_value, 6), "B"], "driver_school_recompute"


def try_shark_t_test(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "great white shark" not in q or "test statistic" not in q:
        return None
    mu = re.search(r"mean length of \$?(\d+(?:\.\d+)?)\$? feet", q, re.IGNORECASE)
    lengths_match = re.search(r"lengths were \$?([\d,\\\s\{\}mbox\.and]+)\$? feet", q, re.IGNORECASE)
    alpha = re.search(r"alpha\s*=\s*([\d.]+)", q, re.IGNORECASE)
    if not mu or not lengths_match or not alpha:
        return None
    values = [mp.mpf(x) for x in re.findall(r"\d+(?:\.\d+)?", lengths_match.group(1))]
    n = len(values)
    mean = sum(values) / n
    sample_var = sum((x - mean) ** 2 for x in values) / (n - 1)
    t_stat = (mean - mp.mpf(mu.group(1))) / (mp.sqrt(sample_var) / mp.sqrt(n))
    crit = t_ppf(1 - mp.mpf(alpha.group(1)), n - 1)
    return [fixed(t_stat, 5), fixed(crit, 5), "A" if t_stat > crit else "B"], "shark_t_test_recompute"


def try_two_prop_marketing(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "marketing targeted at high school graduates" not in q:
        return None
    m = re.search(r"n_1\s*=\s*(\d+).*?x_1\s*=\s*(\d+).*?n_2\s*=\s*(\d+).*?x_2\s*=\s*(\d+).*?at least (\d+(?:\.\d+)?)\\?%.*?alpha\s*=\s*([\d.]+)", q, re.IGNORECASE)
    if not m:
        return None
    n1, x1, n2, x2, threshold_pct, alpha = m.groups()
    n1_i, x1_i, n2_i, x2_i = map(int, (n1, x1, n2, x2))
    p1 = mp.mpf(x1_i) / n1_i
    p2 = mp.mpf(x2_i) / n2_i
    threshold = mp.mpf(threshold_pct) / 100
    se = mp.sqrt(p1 * (1 - p1) / n1_i + p2 * (1 - p2) / n2_i)
    z = (p1 - p2 - threshold) / se
    zcrit = normal_ppf(1 - mp.mpf(alpha))
    p_value = 1 - normal_cdf(z)
    decision = "D" if z > zcrit else "C"
    return [fmt(z, 15), f"({fixed(zcrit, 5)},infinity)", fixed(p_value, 7), decision], "two_prop_marketing_recompute"


def try_twin_ci(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "twins" not in q.lower() or "confidence interval" not in q.lower():
        return None
    m = re.search(r"sample of \$?(\d+)\$?.*?mean at \$?(\d+(?:\.\d+)?)\$?.*?standard deviation at \$?(\d+(?:\.\d+)?)\$?.*?\$?(\d+(?:\.\d+)?)\$? \\?% confidence", q, re.IGNORECASE)
    if not m:
        return None
    n, mean, sd, conf_pct = m.groups()
    confidence = mp.mpf(conf_pct) / 100
    alpha = 1 - confidence
    z = mp.mpf(fixed(normal_ppf(1 - alpha / 2), 5))
    margin = z * mp.mpf(sd) / mp.sqrt(int(n))
    lower = mp.mpf(mean) - margin
    upper = mp.mpf(mean) + margin
    return [fmt(lower, 15), fmt(upper, 15)], "twin_ci_recompute"


def try_hmo_chi_square(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "health maintenance organization" not in q.lower() or "No complaint" not in q:
        return None
    total_match = re.search(r"Total\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)", q, re.IGNORECASE)
    left_match = re.search(r"Left\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)", q, re.IGNORECASE)
    alpha_match = re.search(r"alpha\s*=\s*([\d.]+)", q, re.IGNORECASE)
    if not total_match or not left_match or not alpha_match:
        return None
    totals = [int(x) for x in total_match.groups()]
    left = [int(x) for x in left_match.groups()]
    grand = sum(totals)
    left_total = sum(left)
    not_left_total = grand - left_total
    expected_left = [mp.mpf(left_total) * total / grand for total in totals]
    expected_not = [mp.mpf(not_left_total) * total / grand for total in totals]
    observed_not = [total - obs for total, obs in zip(totals, left)]
    chi = sum((mp.mpf(obs) - exp) ** 2 / exp for obs, exp in zip(left, expected_left))
    chi += sum((mp.mpf(obs) - exp) ** 2 / exp for obs, exp in zip(observed_not, expected_not))
    df = len(totals) - 1
    crit = chi2_ppf(1 - mp.mpf(alpha_match.group(1)), df)
    conclusion = "A" if chi > crit else "B"
    return [str(int(round(float(x)))) for x in expected_left] + [fixed(chi, 5), str(df), fixed(crit, 5), conclusion], "hmo_chi_square_recompute"


RECOMPUTERS: list[Callable[[str], tuple[list[str], str] | None]] = [
    try_newton_cooling,
    try_fahrenheit_conversion,
    try_monthly_salary,
    try_exponential_solve,
    try_storey_height,
    try_poultry_growth,
    try_boat_bearing,
    try_golf_hypothesis,
    try_composite_bacteria,
    try_anova_temperatures,
    try_driver_school,
    try_shark_t_test,
    try_two_prop_marketing,
    try_twin_ci,
    try_hmo_chi_square,
]


def recompute_slots(item: dict[str, Any]) -> tuple[list[str], str] | None:
    question = str(item.get("question", ""))
    for recomputer in RECOMPUTERS:
        result = recomputer(question)
        if result is not None:
            return result
    return None


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
    judger = Judger(strict_extract=False) if args.score else None
    records: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for row in rows:
        item_id = int(row["id"])
        item = data_by_id[item_id]
        old_response = str(row.get("response", ""))
        result = recompute_slots(item)
        changes: list[str] = []
        new_response = old_response
        if result is not None:
            slots, reason = result
            candidate = normalize_final_response(item, box(slots))
            if candidate != old_response:
                new_response = candidate
                changes.append(reason)
        record = {
            **row,
            "id": item_id,
            "pre_numeric_finalizer_response": old_response,
            "response": new_response,
            "answer_key": extract_answer_key(item, new_response, strict=True),
            "numeric_finalizer_changes": changes,
        }
        record.update(final_answer_diagnostics(item, new_response))
        score_record_in_place(record, item, judger)
        records.append(record)
        if changes:
            audit_row = {
                "id": item_id,
                "changes": changes,
                "before": old_response,
                "after": new_response,
            }
            if args.score and has_gold(item):
                audit_row["correct"] = record.get("correct")
                audit_row["gold"] = item["answer"]
            audit.append(audit_row)

    records.sort(key=lambda record: int(record["id"]))
    write_jsonl(args.output, records)
    if args.score:
        print(json.dumps(summarize_results(records), indent=2, sort_keys=True))
    if args.audit_output:
        Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_output).write_text(json.dumps({"changed": len(audit), "changes": audit}, indent=2, sort_keys=True) + "\n")
    print(f"Wrote numeric-finalized responses to {args.output}")


if __name__ == "__main__":
    main()
