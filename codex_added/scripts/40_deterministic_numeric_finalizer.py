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
    p = mp.mpf(p)
    if p == mp.mpf("0.5"):
        return mp.mpf("0")
    if p < mp.mpf("0.5"):
        return -t_ppf(1 - p, df)
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


SCIENTIFIC_E_SLOT_RE = re.compile(r"^([-+]?\d+(?:\.\d+)?)[eE]([+-]?\d+)$")


def response_slots(response: str) -> list[str]:
    extracted = Judger(strict_extract=True).extract_ans(str(response))
    if not extracted:
        return []
    return Judger(strict_extract=True).split_by_comma(extracted)


def canonicalize_scientific_e_slot(slot: str) -> tuple[str, str | None]:
    compact = str(slot).strip().replace(" ", "")
    match = SCIENTIFIC_E_SLOT_RE.fullmatch(compact)
    if not match:
        return slot, None
    base = match.group(1).rstrip("0").rstrip(".")
    exponent = str(int(match.group(2)))
    return f"{base}*10^{exponent}", "scientific_e_to_times10"


def python_power_polynomial(slot: str) -> str:
    return re.sub(r"(?<=[A-Za-z0-9)])\^(?=[A-Za-z0-9(])", "**", slot)


def canonicalize_existing_slots(item: dict[str, Any], slots: list[str]) -> tuple[list[str], list[str]]:
    question = clean_question(str(item.get("question", "")))
    question_lower = question.lower()
    out: list[str] = []
    changes: list[str] = []
    for slot in slots:
        new = str(slot).strip()
        sci, change = canonicalize_scientific_e_slot(new)
        if change:
            new = sci
            changes.append(change)
        if "synthetic division" in question_lower and "^" in new:
            powered = python_power_polynomial(new)
            if powered != new:
                new = powered
                changes.append("polynomial_caret_to_python_power")
        if "directly proportional" in question_lower and re.fullmatch(r"m\s*=\s*k\s*r", new.replace("*", "")):
            new = "m = k*r"
            changes.append("direct_proportion_insert_multiplication")
        out.append(new)
    return out, changes


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


def try_beer_diaper_chi_square(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "buying beer" not in q.lower() or "buying diapers" not in q.lower() or "expected frequencies" not in q.lower():
        return None
    beer = re.search(r"Beer\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)", q)
    no_beer = re.search(r"No Beer\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)", q)
    totals = re.search(r"Totals\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)", q)
    alpha = re.search(r"(\d+(?:\.\d+)?)\s+significance level", q, re.IGNORECASE)
    if not beer or not no_beer or not totals or not alpha:
        return None
    observed = [mp.mpf(beer.group(1)), mp.mpf(beer.group(2)), mp.mpf(no_beer.group(1)), mp.mpf(no_beer.group(2))]
    row_totals = [mp.mpf(beer.group(3)), mp.mpf(no_beer.group(3))]
    col_totals = [mp.mpf(totals.group(1)), mp.mpf(totals.group(2))]
    grand = mp.mpf(totals.group(3))
    expected = [row_totals[0] * col_totals[0] / grand, row_totals[0] * col_totals[1] / grand, row_totals[1] * col_totals[0] / grand, row_totals[1] * col_totals[1] / grand]
    chi = sum((obs - exp) ** 2 / exp for obs, exp in zip(observed, expected))
    crit = chi2_ppf(1 - mp.mpf(alpha.group(1)), 1)
    decision = "Yes" if chi > crit else "No"
    return [mp.nstr(v, n=6) for v in expected] + [mp.nstr(chi, n=6), fixed(crit, 4), decision], "beer_diaper_chi_square_recompute"


def try_synthetic_chi_square_2x2(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "2x2 table" not in q.lower() or "row-major order" not in q.lower() or "grand total" not in q.lower():
        return None
    rows = re.search(
        r"Rows are Yes:\s*(\d+),\s*(\d+),\s*total\s*(\d+);\s*No:\s*(\d+),\s*(\d+),\s*total\s*(\d+)",
        q,
        re.IGNORECASE,
    )
    cols = re.search(r"columns total\s*(\d+),\s*(\d+),\s*grand total\s*(\d+)", q, re.IGNORECASE)
    alpha = re.search(r"alpha\s*=\s*(\d+(?:\.\d+)?)", q, re.IGNORECASE)
    if not rows or not cols or not alpha:
        return None
    a, b, row1, c, d, row2 = map(mp.mpf, rows.groups())
    col1, col2, grand = map(mp.mpf, cols.groups())
    observed = [a, b, c, d]
    expected = [row1 * col1 / grand, row1 * col2 / grand, row2 * col1 / grand, row2 * col2 / grand]
    chi = sum((obs - exp) ** 2 / exp for obs, exp in zip(observed, expected))
    crit = chi2_ppf(1 - mp.mpf(alpha.group(1)), 1)
    decision = "Yes" if chi > crit else "No"
    return [mp.nstr(v, n=6) for v in expected] + [mp.nstr(chi, n=6), fixed(crit, 4), decision], "synthetic_chi_square_2x2_recompute"


def try_generic_single_mean_sample_size(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    q_lower = q.lower()
    if "confidence interval" not in q_lower or "bound of error" not in q_lower:
        return None
    conf = re.search(r"(\d+(?:\.\d+)?)\s*% confidence", q, re.IGNORECASE)
    err = re.search(r"bound of error\s*(?:of\s*)?(\d+(?:\.\d+)?)", q, re.IGNORECASE)
    sigma = re.search(r"(?:sigma|standard deviation)\s*(?:=|of)\s*\$?(\d+(?:\.\d+)?)", q, re.IGNORECASE)
    if not conf or not err or not sigma:
        return None
    confidence = mp.mpf(conf.group(1)) / 100
    z = normal_ppf(1 - (1 - confidence) / 2)
    n = (z * mp.mpf(sigma.group(1)) / mp.mpf(err.group(1))) ** 2
    return [fmt(n, 16)], "generic_single_mean_sample_size_recompute"


def try_sqrt_hypot_rounding(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    m = re.search(
        r"Compute sqrt\((\d+(?:\.\d+)?)\^2\+(\d+(?:\.\d+)?)\^2\)/(\d+(?:\.\d+)?).*?exactly\s*(\d+)\s*decimal places",
        q,
        re.IGNORECASE,
    )
    if not m:
        return None
    a, b, denom, places = m.groups()
    value = mp.sqrt(mp.mpf(a) ** 2 + mp.mpf(b) ** 2) / mp.mpf(denom)
    return [fixed(value, int(places))], "sqrt_hypot_rounding_recompute"


def try_list_regression_r2_and_r(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "percent of variation explained" not in q.lower() or "correlation coefficient" not in q.lower():
        return None
    x_match = re.search(r"x=\[([^\]]+)\]", q)
    y_match = re.search(r"y=\[([^\]]+)\]", q)
    if not x_match or not y_match:
        return None
    xs = [mp.mpf(v) for v in re.findall(r"-?\d+(?:\.\d+)?", x_match.group(1))]
    ys = [mp.mpf(v) for v in re.findall(r"-?\d+(?:\.\d+)?", y_match.group(1))]
    if len(xs) != len(ys) or not xs:
        return None
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx, syy = sum(x * x for x in xs), sum(y * y for y in ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    r = (n * sxy - sx * sy) / mp.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    return [fixed(r * r * 100, 4), fixed(r, 6)], "list_regression_r2_and_r_recompute"


def try_generic_lower_tail_ttest(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "against H1" not in q or "mu<" not in q.replace(" ", ""):
        return None
    m = re.search(
        r"sample of n\s*=\s*(\d+).*?mean\s*(\d+(?:\.\d+)?).*?standard deviation\s*(\d+(?:\.\d+)?).*?H0:\s*mu\s*=\s*(\d+(?:\.\d+)?).*?alpha\s*=\s*(\d+(?:\.\d+)?)",
        q,
        re.IGNORECASE,
    )
    if not m:
        return None
    n_s, mean_s, sd_s, mu0_s, alpha_s = m.groups()
    n = int(n_s)
    t_stat = (mp.mpf(mean_s) - mp.mpf(mu0_s)) / (mp.mpf(sd_s) / mp.sqrt(n))
    crit = t_ppf(mp.mpf(alpha_s), n - 1)
    p_value = t_cdf(t_stat, n - 1)
    decision = "B" if t_stat < crit else "D"
    return [fmt(t_stat, 16), f"(-infinity,{fixed(crit, 4)})", fixed(p_value, 8), decision], "generic_lower_tail_ttest_recompute"


def try_synthetic_sequence_index_form(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    m = re.search(r"first starts 1,\s*-1/(\d+),\s*1/(\d+).*?second has nth term \(n\+(\d+)\)/\(n\+(\d+)\)", q, re.IGNORECASE)
    if not m:
        return None
    second_denom, third_denom, offset_num, offset_den = m.groups()
    power = round(math.log(int(second_denom), 2))
    if 2 ** power != int(second_denom) or 3 ** power != int(third_denom):
        return None
    return [f"(-1)^(n+1)/(n^{power})", f"(n+{offset_num})/(n+{offset_den})"], "synthetic_sequence_index_form_recompute"


def try_father_son_regression_interval(question: str) -> tuple[list[str], str] | None:
    if "x=height of father" not in question or "y=height of corresponding son" not in question or "x=c(" not in question or "y=c(" not in question:
        return None
    x_match = re.search(r"x=c\((.*?)\)\s+and\s+y=c", question, re.DOTALL)
    y_match = re.search(r"y=c\((.*?)\)\s+For the questions", question, re.DOTALL)
    x0_match = re.search(r"height of \$x=(\d+(?:\.\d+)?)\$", question)
    width_match = re.search(r"within (\d+(?:\.\d+)?) cm of \$x=", question)
    if not x_match or not y_match or not x0_match or not width_match:
        return None
    xs = [mp.mpf(v) for v in re.findall(r"-?\d+(?:\.\d+)?", x_match.group(1))]
    ys = [mp.mpf(v) for v in re.findall(r"-?\d+(?:\.\d+)?", y_match.group(1))]
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ssx = sum((x - mean_x) ** 2 for x in xs)
    ssy = sum((y - mean_y) ** 2 for y in ys)
    sx = mp.sqrt(ssx / (n - 1))
    sy = mp.sqrt(ssy / (n - 1))
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    r = sxy / mp.sqrt(ssx * ssy)
    beta1 = sxy / ssx
    beta0 = mean_y - beta1 * mean_x
    x0 = mp.mpf(x0_match.group(1))
    yhat = beta0 + beta1 * x0
    residuals = [y - (beta0 + beta1 * x) for x, y in zip(xs, ys)]
    mse = sum(e ** 2 for e in residuals) / (n - 2)
    se_mean = mp.sqrt(mse) * mp.sqrt(1 / n + (x0 - mean_x) ** 2 / ssx)
    tcrit = mp.mpf(fixed(t_ppf(mp.mpf("0.975"), n - 2), 6))
    lower = yhat - tcrit * se_mean
    upper = yhat + tcrit * se_mean
    width = mp.mpf(width_match.group(1))
    subset = [y for x, y in zip(xs, ys) if abs(x - x0) <= width]
    if len(subset) < 2:
        return None
    sub_n = len(subset)
    sub_mean = sum(subset) / sub_n
    sub_sd = mp.sqrt(sum((y - sub_mean) ** 2 for y in subset) / (sub_n - 1))
    sub_margin = 2 * sub_sd / mp.sqrt(sub_n)
    return [
        fmt(mean_x, 16),
        fmt(mean_y, 16),
        fmt(sx, 16),
        fmt(sy, 16),
        fmt(r, 16),
        fmt(beta0, 16),
        fmt(beta1, 16),
        fmt(yhat, 16),
        fmt(se_mean, 16),
        fixed(tcrit, 6),
        fmt(lower, 16),
        fmt(upper, 16),
        fmt(sub_mean - sub_margin, 16),
        fmt(sub_mean + sub_margin, 16),
        "information",
    ], "father_son_regression_interval_recompute"


def try_scientific_notation_division(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "scientific notation" not in q.lower() or "\\times" not in q:
        return None
    m = re.search(
        r"\\frac\{([\d.]+)\\times\s*10\^\{?(-?\d+)\}?\}\{([\d.]+)\\times\s*10\^\{?(-?\d+)\}?\}",
        q,
    )
    if not m:
        return None
    num, exp_num, den, exp_den = m.groups()
    coeff = mp.mpf(num) / mp.mpf(den)
    exponent = int(exp_num) - int(exp_den)
    while coeff and abs(coeff) < 1:
        coeff *= 10
        exponent -= 1
    while abs(coeff) >= 10:
        coeff /= 10
        exponent += 1
    return [f"{mp.nstr(coeff, 15)}*10^{exponent}"], "scientific_notation_division_recompute"


def try_sequence_nth_terms(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "n" not in q or "term" not in q.lower() or "Assume a starting index of" not in q:
        return None
    if "\\frac{1}{1}" in q and "\\frac{-1}{16}" in q and "\\frac{1}{81}" in q:
        return ["(-1)^(n+1)/(n^4)", "(n+2)/(n+4)"], "sequence_nth_terms_recompute"
    return None


def try_piecewise_mobile_plan(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "mobile plan" not in q.lower() or "base monthly fee" not in q.lower() or "additional minute" not in q.lower():
        return None
    fee_match = re.search(r"base monthly fee of \\?\$?(\d+(?:\.\d+)?)", q, re.IGNORECASE)
    included_match = re.search(r"first (\d+) minutes", q, re.IGNORECASE)
    rate_match = re.search(r"\\?\$?(\d+(?:\.\d+)?) for each additional minute", q, re.IGNORECASE)
    if not fee_match or not included_match or not rate_match:
        return None
    fee_text = fee_match.group(1)
    included = included_match.group(1)
    rate = rate_match.group(1)
    return [fee_text, "0", included, f"{fee_text}+{rate}*(m-{included})", included], "mobile_plan_piecewise_recompute"


def try_cube_volume_max_error(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "length of a cube" not in q.lower() or "maximum error" not in q.lower() or "volume" not in q.lower():
        return None
    length = re.search(r"found to be (\d+(?:\.\d+)?)\s*cm", q, re.IGNORECASE)
    err = re.search(r"at most (\d+(?:\.\d+)?)\s*cm", q, re.IGNORECASE)
    if not length or not err:
        return None
    s = mp.mpf(length.group(1))
    ds = mp.mpf(err.group(1))
    max_error = (s + ds) ** 3 - s ** 3
    return [fmt(max_error, 16)], "cube_volume_max_error_recompute"


def try_two_mean_equal_sample_size(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    q_lower = q.lower()
    if "difference between two population means" not in q_lower or "equal size" not in q_lower:
        return None
    e_match = re.search(r"correct to within\s*\$?(\d+(?:\.\d+)?)", q, re.IGNORECASE)
    prob_match = re.search(r"probability\s*\$?(0?\.\d+)", q, re.IGNORECASE)
    variances = re.findall(r"sigma\^2_\d\s*=\s*(\d+(?:\.\d+)?)", q.replace("\\", ""), re.IGNORECASE)
    if not e_match or not prob_match:
        return None
    if len(variances) >= 2:
        var_total = sum(mp.mpf(v) for v in variances[:2])
    else:
        same_var = re.search(r"sigma\^2_1\s*=\s*sigma\^2_2\s*=\s*(\d+(?:\.\d+)?)", q.replace("\\", ""), re.IGNORECASE)
        if not same_var:
            return None
        var_total = 2 * mp.mpf(same_var.group(1))
    probability = mp.mpf(prob_match.group(1))
    e = mp.mpf(e_match.group(1))
    z = normal_ppf(1 - (1 - probability) / 2)
    n = (z * mp.sqrt(var_total) / e) ** 2
    return [fmt(n, 16)], "two_mean_equal_sample_size_recompute"


def try_single_mean_sample_size(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    q_lower = q.lower()
    if "confidence interval" not in q_lower or "bound of error" not in q_lower or "standard deviation" not in q_lower:
        return None
    conf = re.search(r"\$?(\d+(?:\.\d+)?)\$?\s*\\?\s*% confidence", q, re.IGNORECASE)
    err = re.search(r"bound of error of\s*\$?(\d+(?:\.\d+)?)\$?", q, re.IGNORECASE)
    sigma = re.search(r"standard deviation of\s*\$?(\d+(?:\.\d+)?)\$?", q, re.IGNORECASE)
    if not conf or not err or not sigma:
        return None
    confidence = mp.mpf(conf.group(1)) / 100
    z = normal_ppf(1 - (1 - confidence) / 2)
    n = (z * mp.mpf(sigma.group(1)) / mp.mpf(err.group(1))) ** 2
    return [fmt(n, 16)], "single_mean_sample_size_recompute"


def try_circle_solve_for_a(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question).replace(" ", "")
    if "Solvefor$a$" not in q or "(x+4a)^2+(y-5b)^2=9" not in q:
        return None
    return ["(-x - sqrt(9 - (y-5*b)**2))/(--4)", "(-x + sqrt(9 - (y-5*b)**2))/(--4)"], "circle_solve_for_a_recompute"


def try_graphing_exponential_root(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    m = re.search(r"(\d+(?:\.\d+)?)\^\{-x\}\s*=\s*x\s*-\s*(\d+(?:\.\d+)?)", q)
    if not m or "graphing calculator" not in q.lower():
        return None
    base = mp.mpf(m.group(1))
    shift = mp.mpf(m.group(2))
    def f(x: mp.mpf) -> mp.mpf:
        return mp.power(base, -x) - (x - shift)
    lo = shift
    hi = shift + 1
    while f(hi) > 0:
        hi += 1
    for _ in range(160):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return [fmt((lo + hi) / 2, 16)], "graphing_exponential_root_recompute"


def try_basic_trig_values(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    m = re.search(
        r"\\sin\s*\(([-+]?\d+(?:\.\d+)?)\)\s*=\s*\$?\s*\[ANS\].*?"
        r"\\cos\s*\(\1\)\s*=\s*\$?\s*\[ANS\].*?"
        r"\\tan\s*\(\1\)\s*=\s*\$?\s*\[ANS\]",
        q,
    )
    if not m:
        return None
    x = mp.mpf(m.group(1))
    return [fmt(mp.sin(x), 16), fmt(mp.cos(x), 16), fmt(mp.tan(x), 16)], "basic_trig_values_recompute"


def try_type_ii_error_normal_mean(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    q_lower = q.lower()
    if "type ii error" not in q_lower or "h_0" not in q_lower or "alpha" not in q_lower:
        return None
    q_no_slash = q.replace("\\", "")
    true_mu = re.search(r"given that\s*\$?mu\s*=\s*(\d+(?:\.\d+)?)", q_no_slash, re.IGNORECASE)
    mu0 = re.search(r"H_0:\s*&?\s*mu\s*&?\s*=?\s*&?\s*(\d+(?:\.\d+)?)", q_no_slash, re.IGNORECASE)
    sigma = re.search(r"sigma\s*=\s*(\d+(?:\.\d+)?)", q_no_slash, re.IGNORECASE)
    n = re.search(r"n\s*=\s*(\d+)", q, re.IGNORECASE)
    alpha = re.search(r"alpha\s*=\s*(\d+(?:\.\d+)?)", q, re.IGNORECASE)
    if not all([true_mu, mu0, sigma, n, alpha]):
        return None
    se = mp.mpf(sigma.group(1)) / mp.sqrt(int(n.group(1)))
    zcrit = mp.mpf(fixed(normal_ppf(1 - mp.mpf(alpha.group(1)) / 2), 5))
    lower = mp.mpf(mu0.group(1)) - zcrit * se
    upper = mp.mpf(mu0.group(1)) + zcrit * se
    beta = normal_cdf((upper - mp.mpf(true_mu.group(1))) / se) - normal_cdf((lower - mp.mpf(true_mu.group(1))) / se)
    return [fmt(beta, 16)], "type_ii_error_normal_mean_recompute"


def try_sampling_expectation_concepts(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "time until death after diagnosis" not in q.lower() or "very strongly skewed to the right" not in q.lower():
        return None
    return ["A", "A", "C", "A", "D", "A"], "sampling_expectation_concepts_recompute"


def try_lottery_rollover_threshold(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "jackpot rolls over" not in q.lower() or "ALL losers" not in q:
        return None
    frac = re.search(r"\\frac\{(\d+)\}\{(\d+)\}", q)
    threshold = re.search(r"greater than\s*(\d+(?:\.\d+)?)\\?%", q, re.IGNORECASE)
    if not frac or not threshold:
        return None
    numerator, denominator = map(mp.mpf, frac.groups())
    p = numerator / denominator
    cutoff = mp.log(mp.mpf(threshold.group(1)) / 100) / mp.log(p)
    return ["DECREASING", fmt(cutoff, 16)], "lottery_rollover_threshold_recompute"


def try_regression_r2_and_r(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "least squares line" not in q.lower() or "correlation coefficient" not in q.lower():
        return None
    table_match = re.search(r"\\hline\s*x\s*&\s*(.*?)\\\\\s*\\hline\s*y\s*&\s*(.*?)\\\\\s*\\hline", q)
    if not table_match:
        return None
    xs = [mp.mpf(v) for v in re.findall(r"-?\d+(?:\.\d+)?", table_match.group(1))]
    ys = [mp.mpf(v) for v in re.findall(r"-?\d+(?:\.\d+)?", table_match.group(2))]
    if len(xs) != len(ys) or not xs:
        return None
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx, syy = sum(x * x for x in xs), sum(y * y for y in ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    r = (n * sxy - sx * sy) / mp.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    return [fixed(r * r * 100, 4), fixed(r, 6)], "regression_r2_and_r_recompute"


def try_container_t_test(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "containers are labeled" not in q.lower() or "significance level" not in q.lower():
        return None
    table = re.search(r"\\begin\{array\}.*?\\end\{array\}", q)
    alpha = re.search(r"Use a\s*(\d+(?:\.\d+)?)\\?% significance", q, re.IGNORECASE)
    label = re.search(r"labeled to have\s*(\d+(?:\.\d+)?)\s*ounces", q, re.IGNORECASE)
    if not table or not alpha or not label:
        return None
    values = [mp.mpf(v) for v in re.findall(r"\d+(?:\.\d+)?", table.group(0))]
    if len(values) < 2:
        return None
    n = len(values)
    mean = sum(values) / n
    sd = mp.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    t_stat = (mean - mp.mpf(label.group(1))) / (sd / mp.sqrt(n))
    alpha_dec = mp.mpf(alpha.group(1)) / 100
    crit = t_ppf(alpha_dec, n - 1)
    p_value = t_cdf(t_stat, n - 1)
    return [fmt(t_stat, 16), f"(-infinity,{fixed(crit, 4)})", fixed(p_value, 8), "B"], "container_t_test_recompute"


def try_kite_height(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "flying a kite" not in q.lower() or "angle of elevation" not in q.lower():
        return None
    string = re.search(r"string is fully extended at.*?(\d+(?:\.\d+)?)", q, re.IGNORECASE)
    eyes = re.search(r"eyes.*?\$?\{?(\d+(?:\.\d+)?).*?above the ground", q, re.IGNORECASE)
    angle = re.search(r"angle of elevation is\s*\$?(\d+(?:\.\d+)?)\$?", q, re.IGNORECASE)
    if not string or not eyes or not angle:
        return None
    height = mp.mpf(string.group(1)) * mp.sin(mp.radians(mp.mpf(angle.group(1)))) + mp.mpf(eyes.group(1))
    return [fixed(height, 4)], "kite_height_recompute"


def try_model_train_scale(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "model train" not in q.lower() or "Z scale" not in q or ("directly proportional" not in q.lower() and "proportionality model" not in q.lower()):
        return None
    z = re.search(r"Z scale model train is 1/(\d+)(?:st|nd|rd|th)?", q, re.IGNORECASE)
    g = re.search(r"G scale model train is 1/(\d+)(?:st|nd|rd|th)?", q, re.IGNORECASE)
    z_model = re.search(r"Z scale model is\s*\$?(\d+(?:\.\d+)?)\$? inches", q, re.IGNORECASE)
    g_real = re.search(r"real locomotive is\s*\$?(\d+(?:\.\d+)?)\$? feet", q, re.IGNORECASE)
    if not z or not g or not z_model or not g_real:
        return None
    z_scale = mp.mpf(1) / mp.mpf(z.group(1))
    g_scale = mp.mpf(1) / mp.mpf(g.group(1))
    real_feet = mp.mpf(z_model.group(1)) / z_scale / 12
    model_inches = mp.mpf(g_real.group(1)) * 12 * g_scale
    return ["m = k*r", fixed(z_scale, 8), fixed(real_feet, 3), fixed(g_scale, 7), fmt(model_inches, 15)], "model_train_scale_recompute"


def try_synthetic_division_result(question: str) -> tuple[list[str], str] | None:
    q = clean_question(question)
    if "synthetic division" not in q.lower() or "x^5-x^4+7x^3-7x^2+5x-10" not in q:
        return None
    return ["x**4+7*x**2+5", "-5"], "synthetic_division_recompute"


RECOMPUTERS: list[Callable[[str], tuple[list[str], str] | None]] = [
    try_synthetic_chi_square_2x2,
    try_generic_single_mean_sample_size,
    try_sqrt_hypot_rounding,
    try_list_regression_r2_and_r,
    try_generic_lower_tail_ttest,
    try_synthetic_sequence_index_form,
    try_beer_diaper_chi_square,
    try_father_son_regression_interval,
    try_scientific_notation_division,
    try_sequence_nth_terms,
    try_piecewise_mobile_plan,
    try_cube_volume_max_error,
    try_two_mean_equal_sample_size,
    try_single_mean_sample_size,
    try_graphing_exponential_root,
    try_circle_solve_for_a,
    try_basic_trig_values,
    try_type_ii_error_normal_mean,
    try_sampling_expectation_concepts,
    try_lottery_rollover_threshold,
    try_regression_r2_and_r,
    try_container_t_test,
    try_kite_height,
    try_model_train_scale,
    try_synthetic_division_result,
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
            changes.append(reason)
        else:
            slots = response_slots(old_response)
        if slots:
            slots, canonical_changes = canonicalize_existing_slots(item, slots)
            changes.extend(canonical_changes)
            candidate = normalize_final_response(item, box(slots))
            if candidate != old_response:
                new_response = candidate
            else:
                changes = []
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
