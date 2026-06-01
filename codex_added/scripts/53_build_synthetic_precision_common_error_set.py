from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mpmath as mp

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from judger import Judger
from math_comp.data import write_jsonl
from math_comp.scoring import score_item

mp.mp.dps = 50
SEED = 1516


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic precision/rounding/common-error eval fixtures.")
    parser.add_argument(
        "--output-prefix",
        default="codex_added/job_data/synthetic_precision_common_errors_v2",
        help="Output prefix without extension.",
    )
    parser.add_argument("--start-id", type=int, default=940000)
    parser.add_argument("--cases-per-category", type=int, default=30, help="Minimum rows to generate for each synthetic category.")
    return parser.parse_args()


def fmt(value: mp.mpf | float | int, digits: int = 16) -> str:
    return mp.nstr(mp.mpf(value), n=digits, strip_zeros=False)


def fixed(value: mp.mpf | float | int, places: int) -> str:
    return f"{float(value):.{places}f}"


def box(slots: list[str]) -> str:
    return "\\boxed{" + ", ".join(str(slot).strip() for slot in slots) + "}"


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


class Builder:
    def __init__(self, start_id: int):
        self.next_id = start_id
        self.items: list[dict[str, Any]] = []
        self.gold_predictions: list[dict[str, Any]] = []
        self.bad_predictions: list[dict[str, Any]] = []
        self.finalizer_pairs: list[dict[str, Any]] = []

    def add(
        self,
        category: str,
        target_error: str,
        question: str,
        answer: list[str],
        bad_slots: list[str],
        notes: str,
    ) -> None:
        item_id = self.next_id
        self.next_id += 1
        gold_response = box(answer)
        bad_response = box(bad_slots)
        item = {
            "id": item_id,
            "question": question,
            "answer": answer,
            "synthetic_category": category,
            "target_error": target_error,
            "notes": notes,
        }
        self.items.append(item)
        self.gold_predictions.append(
            {
                "id": item_id,
                "response": gold_response,
                "is_mcq": False,
                "synthetic_category": category,
                "target_error": target_error,
            }
        )
        self.bad_predictions.append(
            {
                "id": item_id,
                "response": bad_response,
                "is_mcq": False,
                "synthetic_category": category,
                "target_error": target_error,
            }
        )
        self.finalizer_pairs.append(
            {
                "id": item_id,
                "synthetic_category": category,
                "target_error": target_error,
                "question": question,
                "source_response": f"Reasoning omitted. Final answer: {bad_response}",
                "target_response": gold_response,
                "answer": answer,
            }
        )


def add_scientific_notation(builder: Builder) -> None:
    cases = [
        ("4.5", 4, "9", -3),
        ("7.2", 8, "3", 2),
        ("9.6", -2, "1.2", -7),
        ("3.5", 6, "7", -1),
        ("8.1", 5, "2.7", -4),
        ("6.4", -3, "8", -8),
        ("5.5", 9, "1.1", 3),
        ("2.4", 1, "6", -5),
    ]
    for num, exp_num, den, exp_den in cases:
        coeff = mp.mpf(num) / mp.mpf(den)
        exponent = exp_num - exp_den
        while coeff and abs(coeff) < 1:
            coeff *= 10
            exponent -= 1
        while abs(coeff) >= 10:
            coeff /= 10
            exponent += 1
        base = mp.nstr(coeff, 12).rstrip("0").rstrip(".")
        answer = [f"{base}*10^{exponent}"]
        bad = [f"{base}E{exponent}"]
        question = (
            "Divide the following numbers, writing your answer in scientific notation.\n"
            f"$\\frac{{{num}\\times 10^{{{exp_num}}}}}{{{den}\\times 10^{{{exp_den}}}}}=$ [ANS]"
        )
        builder.add("scientific_notation", "E notation instead of scorer-safe *10^ form", question, answer, bad, "Tests E notation canonicalization.")


def add_sample_size(builder: Builder) -> None:
    cases = [
        (90, "3.8", "1.1"),
        (95, "4.9", "2"),
        (98, "7.2", "1.5"),
        (99, "12.5", "3"),
        (92, "5.4", "1.7"),
        (96, "8.8", "2.4"),
        (94, "6.1", "1.3"),
        (97, "10.2", "2.1"),
    ]
    for confidence_pct, sigma, error in cases:
        confidence = mp.mpf(confidence_pct) / 100
        z = normal_ppf(1 - (1 - confidence) / 2)
        n = (z * mp.mpf(sigma) / mp.mpf(error)) ** 2
        answer = [fmt(n, 16)]
        bad = [str(math.ceil(float(n)))]
        question = (
            f"A pilot study suggests sigma={sigma}. For a {confidence_pct}% confidence interval "
            f"with bound of error {error}, compute the real-valued sample size expression before any integer ceiling. "
            "Do not round up. n=[ANS]"
        )
        builder.add("sample_size_precision", "ceiled integer instead of unrounded real value", question, answer, bad, "Tests premature integer rounding.")


def add_rounding_prompt(builder: Builder) -> None:
    cases = [
        ("13.7", "2.6", 4),
        ("21.3", "4.7", 4),
        ("8.4", "1.9", 4),
        ("55.2", "7.1", 4),
        ("31.5", "3.8", 4),
        ("44.8", "5.9", 4),
        ("17.6", "2.3", 4),
        ("63.4", "8.5", 4),
    ]
    for a, b, places in cases:
        value = mp.sqrt(mp.mpf(a) ** 2 + mp.mpf(b) ** 2) / mp.mpf("3.7")
        answer = [fixed(value, places)]
        bad = [fixed(value, 2)]
        question = f"Compute sqrt({a}^2+{b}^2)/3.7. Round the answer to exactly {places} decimal places. [ANS]"
        builder.add("explicit_rounding_precision", "too few decimal places", question, answer, bad, "Tests honoring explicit decimal-place instructions.")


def add_trig_precision(builder: Builder) -> None:
    for raw_x in ["0.3", "0.45", "0.6", "0.75", "0.9", "1.1", "1.25", "1.4"]:
        x = mp.mpf(raw_x)
        answer = [fmt(mp.sin(x), 16), fmt(mp.cos(x), 16), fmt(mp.tan(x), 16)]
        bad = [fixed(mp.sin(x), 6), fixed(mp.cos(x), 6), fixed(mp.tan(x), 3)]
        question = f"Find the following values using radians: $\\sin ({raw_x})=$ [ANS] ; $\\cos ({raw_x})=$ [ANS] ; $\\tan ({raw_x})=$ [ANS]."
        builder.add("trig_precision", "low precision transcendental values", question, answer, bad, "Tests sin/cos/tan precision.")


def add_chi_square(builder: Builder) -> None:
    cases = [
        (7, 49, 10, 66, "0.01"),
        (12, 31, 8, 45, "0.05"),
        (18, 22, 15, 37, "0.10"),
        (9, 28, 14, 33, "0.05"),
        (25, 41, 19, 58, "0.01"),
        (16, 54, 21, 49, "0.10"),
    ]
    for a, b, c, d, alpha in cases:
        row1, row2 = a + b, c + d
        col1, col2 = a + c, b + d
        total = row1 + row2
        expected = [mp.mpf(row1) * col1 / total, mp.mpf(row1) * col2 / total, mp.mpf(row2) * col1 / total, mp.mpf(row2) * col2 / total]
        observed = [mp.mpf(a), mp.mpf(b), mp.mpf(c), mp.mpf(d)]
        chi = sum((obs - exp) ** 2 / exp for obs, exp in zip(observed, expected))
        crit = chi2_ppf(1 - mp.mpf(alpha), 1)
        decision = "Yes" if chi > crit else "No"
        answer = [mp.nstr(v, n=6) for v in expected] + [mp.nstr(chi, n=6), fixed(crit, 4), decision]
        bad = [fmt(v, 15) for v in expected] + [fmt(chi * mp.mpf("1.0002"), 15), fixed(crit, 4), decision, decision]
        question = (
            f"Test a dependent relationship in this 2x2 table at alpha={alpha}. "
            f"Rows are Yes: {a}, {b}, total {row1}; No: {c}, {d}, total {row2}; columns total {col1}, {col2}, grand total {total}. "
            "Return expected counts in row-major order, the chi-square statistic, the critical value, and Yes or No. "
            "[ANS] [ANS] [ANS] [ANS] [ANS] [ANS] [ANS]"
        )
        builder.add("chi_square_slot_count", "duplicate decision slot and perturbed statistic", question, answer, bad, "Tests duplicate final slots and chi-square precision.")


def add_ttest_lower_tail(builder: Builder) -> None:
    cases = [
        (14, "10.0", "9.78", "0.23", "0.02"),
        (12, "50.0", "49.1", "1.2", "0.05"),
        (18, "100.0", "98.7", "2.4", "0.01"),
        (10, "32.0", "30.9", "1.1", "0.05"),
        (16, "5.0", "4.72", "0.31", "0.10"),
        (20, "75.0", "73.4", "2.8", "0.05"),
    ]
    for n, mu0, mean, sd, alpha in cases:
        df = n - 1
        t_stat = (mp.mpf(mean) - mp.mpf(mu0)) / (mp.mpf(sd) / mp.sqrt(n))
        crit = t_ppf(mp.mpf(alpha), df)
        p_value = t_cdf(t_stat, df)
        decision = "B" if t_stat < crit else "D"
        answer = [fmt(t_stat, 16), f"(-infinity,{fixed(crit, 4)})", fixed(p_value, 8), decision]
        bad = [fixed(t_stat, 3), f"(-infinity,{fixed(-crit, 4)})", fixed(p_value * mp.mpf("1.1"), 4), decision]
        question = (
            f"A sample of n={n} has mean {mean} and sample standard deviation {sd}. "
            f"Test H0: mu={mu0} against H1: mu<{mu0} at alpha={alpha}. "
            "Return the t statistic, the lower-tail rejection region, the p-value, and decision letter B for Reject H0 or D for Do Not Reject H0. "
            "[ANS] [ANS] [ANS] [ANS]"
        )
        builder.add("ttest_lower_tail", "wrong critical-value tail and rounded t statistic", question, answer, bad, "Tests lower-tail t critical values.")


def add_regression_precision(builder: Builder) -> None:
    datasets = [
        ([7, 5, 6, 6, 2, 3, 5], [4, 5, 6, 4, 8, 8, 6]),
        ([1, 2, 3, 4, 5, 6], [2, 4, 5, 7, 9, 11]),
        ([2, 4, 6, 8, 10], [12, 11, 9, 8, 6]),
        ([3, 5, 7, 9, 11, 13], [4, 6, 7, 10, 12, 13]),
        ([10, 12, 14, 16, 18], [25, 23, 20, 18, 15]),
        ([1, 3, 4, 7, 9, 11], [8, 7, 7, 5, 4, 3]),
    ]
    for xs_i, ys_i in datasets:
        xs = [mp.mpf(x) for x in xs_i]
        ys = [mp.mpf(y) for y in ys_i]
        n = len(xs)
        sx, sy = sum(xs), sum(ys)
        sxx, syy = sum(x * x for x in xs), sum(y * y for y in ys)
        sxy = sum(x * y for x, y in zip(xs, ys))
        r = (n * sxy - sx * sy) / mp.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
        answer = [fixed(r * r * 100, 4), fixed(r, 6)]
        bad = [fixed(r * r * 100, 0), fixed(r, 3)]
        question = (
            f"For x={xs_i} and y={ys_i}, report the percent of variation explained by the least-squares line and the correlation coefficient r. "
            "Use 4 decimal places for the percent and 6 decimal places for r. [ANS] [ANS]"
        )
        builder.add("regression_rounding", "rounded R^2 percent and r", question, answer, bad, "Tests regression precision.")


def add_synthetic_division(builder: Builder) -> None:
    cases = [
        ([1, 0, 7, 0, 5], -2, -5),
        ([2, -3, 0, 4], 1, 7),
        ([1, 5, -2], -3, 11),
        ([3, 0, 4, -8], 2, -9),
        ([1, -4, 6, -1], -1, 3),
        ([2, 1, -5], 4, -6),
    ]
    for coeffs, c, remainder in cases:
        degree = len(coeffs) - 1
        terms = []
        for idx, coeff in enumerate(coeffs):
            power = degree - idx
            if coeff == 0:
                continue
            sign = "+" if coeff > 0 and terms else ""
            abs_coeff = abs(coeff)
            if power == 0:
                term = f"{sign}{coeff}"
            elif power == 1:
                term = f"{sign}{'' if abs_coeff == 1 else abs_coeff}x"
                if coeff < 0:
                    term = f"-{'' if abs_coeff == 1 else abs_coeff}x"
            else:
                term = f"{sign}{'' if abs_coeff == 1 else abs_coeff}x^{power}"
                if coeff < 0:
                    term = f"-{'' if abs_coeff == 1 else abs_coeff}x^{power}"
            terms.append(term)
        quotient_display = "".join(terms)
        quotient_gold = quotient_display.replace("x^", "x**").replace("2x", "2*x").replace("3x", "3*x").replace("4x", "4*x").replace("5x", "5*x").replace("6x", "6*x").replace("7x", "7*x")
        quotient_bad = quotient_gold.replace("**", "^")
        divisor = f"x-{c}" if c >= 0 else f"x+{abs(c)}"
        question = f"Using synthetic division, divide a polynomial by {divisor}. The quotient is {quotient_display} and the remainder is requested. Quotient=[ANS] Remainder=[ANS]"
        builder.add("algebra_power_syntax", "caret power syntax where scorer expects Python-style power", question, [quotient_gold, str(remainder)], [quotient_bad, str(remainder)], "Tests ^ versus ** canonicalization.")


def add_direct_proportion(builder: Builder) -> None:
    cases = [
        (220, 24, "40", "550"),
        (160, 32, "48", "420"),
        (87, 22, "36", "210"),
        (64, 16, "12", "80"),
        (120, 30, "24", "360"),
        (200, 25, "60", "500"),
    ]
    for z_den, g_den, z_inches, real_feet in cases:
        z_scale = mp.mpf(1) / z_den
        g_scale = mp.mpf(1) / g_den
        inferred_real = mp.mpf(z_inches) / z_scale / 12
        g_inches = mp.mpf(real_feet) * 12 * g_scale
        answer = ["m = k*r", fixed(z_scale, 8), fixed(inferred_real, 3), fixed(g_scale, 7), fmt(g_inches, 15)]
        bad = ["m = k/r", fixed(z_scale, 3), fixed(inferred_real, 0), fixed(g_scale, 3), fixed(g_inches / 12, 3)]
        question = (
            f"A Z scale model train is 1/{z_den}th of the real size and a G scale model train is 1/{g_den}th of the real size. "
            f"The Z scale model is {z_inches} inches long. A real locomotive is {real_feet} feet long. "
            "Give the proportionality model m=[ANS], Z scale k=[ANS], inferred real length in feet=[ANS], G scale k=[ANS], and G model length in inches=[ANS]."
        )
        builder.add("direct_proportion_form", "implicit multiplication and exact fractions where decimals are required", question, answer, bad, "Tests m=k*r and decimal-form answers.")


def add_sequence_index_form(builder: Builder) -> None:
    for power in [2, 3, 4, 5, 6, 7]:
        answer = [f"(-1)^(n+1)/(n^{power})", f"(n+{power - 1})/(n+{power + 1})"]
        bad = [f"(-1)^n/(n^{power})", f"(n+{power - 1})/(n+{power + 1})"]
        question = (
            f"Find formulas for two sequences. The first starts 1, -1/{2**power}, 1/{3**power}, ... and uses starting index n=1. "
            f"The second has nth term (n+{power - 1})/(n+{power + 1}). First formula=[ANS] Second formula=[ANS]"
        )
        builder.add("sequence_canonical_form", "off-by-one alternating sign convention", question, answer, bad, "Tests sequence sign-index discipline.")


def add_piecewise_literals(builder: Builder) -> None:
    cases = [
        ("25.00", "500", "0.65"),
        ("18.00", "300", "0.40"),
        ("32.00", "750", "0.25"),
        ("12.50", "250", "0.15"),
        ("45.00", "900", "0.35"),
        ("27.00", "450", "0.55"),
    ]
    for base, included, rate in cases:
        answer = [base, "0", included, f"{base}+{rate}*(m-{included})", included]
        bad_base = fmt(mp.mpf(base) + mp.mpf(rate) / 1000, 10)
        bad = [bad_base, "0", included, f"{base}+{rate}*(m-{included})", included]
        question = (
            f"A mobile plan has a base monthly fee of ${base} for the first {included} minutes and charges ${rate} for each additional minute. "
            "Give base fee, initial marginal cost, included minutes, overage formula, and breakpoint. [ANS] [ANS] [ANS] [ANS] [ANS]"
        )
        builder.add("question_literal_precision", "expanded a literal copied from the question", question, answer, bad, "Tests not adding bogus precision to copied constants.")


def category_count(builder: Builder, category: str) -> int:
    return sum(1 for item in builder.items if item["synthetic_category"] == category)


def add_extra_cases(builder: Builder, target_per_category: int) -> None:
    rng = random.Random(SEED + 1000)

    def needs(category: str) -> bool:
        return category_count(builder, category) < target_per_category

    idx = 0
    while needs("scientific_notation"):
        num = f"{rng.randint(12, 99) / 10:.1f}"
        den = f"{rng.choice([12, 15, 18, 21, 24, 27, 32, 36, 42, 48]) / 10:.1f}"
        exp_num = rng.randint(-6, 10)
        exp_den = rng.randint(-8, 7)
        coeff = mp.mpf(num) / mp.mpf(den)
        exponent = exp_num - exp_den
        while coeff and abs(coeff) < 1:
            coeff *= 10
            exponent -= 1
        while abs(coeff) >= 10:
            coeff /= 10
            exponent += 1
        base = mp.nstr(coeff, 12).rstrip("0").rstrip(".")
        question = (
            "Divide the following numbers, writing your answer in scientific notation.\n"
            f"$\\frac{{{num}\\times 10^{{{exp_num}}}}}{{{den}\\times 10^{{{exp_den}}}}}=$ [ANS]"
        )
        builder.add("scientific_notation", "E notation instead of scorer-safe *10^ form", question, [f"{base}*10^{exponent}"], [f"{base}E{exponent}"], "Randomized E notation canonicalization case.")
        idx += 1

    while needs("sample_size_precision"):
        confidence_pct = rng.choice([88, 90, 92, 94, 95, 96, 97, 98, 99])
        sigma = f"{rng.randint(25, 150) / 10:.1f}"
        error = f"{rng.randint(6, 45) / 10:.1f}"
        confidence = mp.mpf(confidence_pct) / 100
        z = normal_ppf(1 - (1 - confidence) / 2)
        n = (z * mp.mpf(sigma) / mp.mpf(error)) ** 2
        question = (
            f"A pilot study suggests sigma={sigma}. For a {confidence_pct}% confidence interval "
            f"with bound of error {error}, compute the real-valued sample size expression before any integer ceiling. "
            "Do not round up. n=[ANS]"
        )
        builder.add("sample_size_precision", "ceiled integer instead of unrounded real value", question, [fmt(n, 16)], [str(math.ceil(float(n)))], "Randomized premature integer rounding case.")

    while needs("explicit_rounding_precision"):
        a = f"{rng.randint(50, 900) / 10:.1f}"
        b = f"{rng.randint(20, 160) / 10:.1f}"
        denom = f"{rng.randint(21, 75) / 10:.1f}"
        places = rng.choice([3, 4, 5])
        value = mp.sqrt(mp.mpf(a) ** 2 + mp.mpf(b) ** 2) / mp.mpf(denom)
        question = f"Compute sqrt({a}^2+{b}^2)/{denom}. Round the answer to exactly {places} decimal places. [ANS]"
        builder.add("explicit_rounding_precision", "too few decimal places", question, [fixed(value, places)], [fixed(value, max(1, places - 2))], "Randomized explicit rounding case.")

    while needs("trig_precision"):
        raw_x = f"{rng.randint(12, 160) / 100:.2f}"
        x = mp.mpf(raw_x)
        question = f"Find the following values using radians: $\\sin ({raw_x})=$ [ANS] ; $\\cos ({raw_x})=$ [ANS] ; $\\tan ({raw_x})=$ [ANS]."
        builder.add("trig_precision", "low precision transcendental values", question, [fmt(mp.sin(x), 16), fmt(mp.cos(x), 16), fmt(mp.tan(x), 16)], [fixed(mp.sin(x), 6), fixed(mp.cos(x), 6), fixed(mp.tan(x), 3)], "Randomized trig precision case.")

    while needs("chi_square_slot_count"):
        a, b, c, d = [rng.randint(5, 85) for _ in range(4)]
        alpha = rng.choice(["0.01", "0.05", "0.10"])
        row1, row2 = a + b, c + d
        col1, col2 = a + c, b + d
        total = row1 + row2
        expected = [mp.mpf(row1) * col1 / total, mp.mpf(row1) * col2 / total, mp.mpf(row2) * col1 / total, mp.mpf(row2) * col2 / total]
        observed = [mp.mpf(a), mp.mpf(b), mp.mpf(c), mp.mpf(d)]
        chi = sum((obs - exp) ** 2 / exp for obs, exp in zip(observed, expected))
        crit = chi2_ppf(1 - mp.mpf(alpha), 1)
        decision = "Yes" if chi > crit else "No"
        question = (
            f"Test a dependent relationship in this 2x2 table at alpha={alpha}. "
            f"Rows are Yes: {a}, {b}, total {row1}; No: {c}, {d}, total {row2}; columns total {col1}, {col2}, grand total {total}. "
            "Return expected counts in row-major order, the chi-square statistic, the critical value, and Yes or No. "
            "[ANS] [ANS] [ANS] [ANS] [ANS] [ANS] [ANS]"
        )
        answer = [mp.nstr(v, n=6) for v in expected] + [mp.nstr(chi, n=6), fixed(crit, 4), decision]
        bad = [fmt(v, 15) for v in expected] + [fmt(chi * mp.mpf("1.0002"), 15), fixed(crit, 4), decision, decision]
        builder.add("chi_square_slot_count", "duplicate decision slot and perturbed statistic", question, answer, bad, "Randomized duplicate-slot chi-square case.")

    while needs("ttest_lower_tail"):
        n = rng.randint(8, 35)
        mu0 = mp.mpf(rng.randint(200, 900)) / 10
        sd = mp.mpf(rng.randint(8, 55)) / 10
        delta = mp.mpf(rng.randint(4, 30)) / 10
        mean = mu0 - delta
        alpha = rng.choice(["0.01", "0.02", "0.05", "0.10"])
        t_stat = (mean - mu0) / (sd / mp.sqrt(n))
        crit = t_ppf(mp.mpf(alpha), n - 1)
        p_value = t_cdf(t_stat, n - 1)
        decision = "B" if t_stat < crit else "D"
        question = (
            f"A sample of n={n} has mean {fixed(mean, 2)} and sample standard deviation {fixed(sd, 2)}. "
            f"Test H0: mu={fixed(mu0, 2)} against H1: mu<{fixed(mu0, 2)} at alpha={alpha}. "
            "Return the t statistic, the lower-tail rejection region, the p-value, and decision letter B for Reject H0 or D for Do Not Reject H0. "
            "[ANS] [ANS] [ANS] [ANS]"
        )
        answer = [fmt(t_stat, 16), f"(-infinity,{fixed(crit, 4)})", fixed(p_value, 8), decision]
        bad = [fixed(t_stat, 3), f"(-infinity,{fixed(-crit, 4)})", fixed(p_value * mp.mpf("1.1"), 4), decision]
        builder.add("ttest_lower_tail", "wrong critical-value tail and rounded t statistic", question, answer, bad, "Randomized lower-tail t-test case.")

    while needs("regression_rounding"):
        n = rng.randint(5, 9)
        xs_i = sorted(rng.sample(range(1, 25), n))
        slope = rng.choice([-3, -2, -1, 1, 2, 3])
        intercept = rng.randint(-5, 12)
        ys_i = [intercept + slope * x + rng.choice([-3, -1, 0, 2, 4]) for x in xs_i]
        xs = [mp.mpf(x) for x in xs_i]
        ys = [mp.mpf(y) for y in ys_i]
        sx, sy = sum(xs), sum(ys)
        sxx, syy = sum(x * x for x in xs), sum(y * y for y in ys)
        sxy = sum(x * y for x, y in zip(xs, ys))
        denom = (n * sxx - sx * sx) * (n * syy - sy * sy)
        if denom == 0:
            continue
        r = (n * sxy - sx * sy) / mp.sqrt(denom)
        question = (
            f"For x={xs_i} and y={ys_i}, report the percent of variation explained by the least-squares line and the correlation coefficient r. "
            "Use 4 decimal places for the percent and 6 decimal places for r. [ANS] [ANS]"
        )
        builder.add("regression_rounding", "rounded R^2 percent and r", question, [fixed(r * r * 100, 4), fixed(r, 6)], [fixed(r * r * 100, 0), fixed(r, 3)], "Randomized regression rounding case.")

    while needs("algebra_power_syntax"):
        degree = rng.randint(2, 5)
        terms = []
        for power in range(degree, -1, -1):
            coeff = rng.choice([c for c in range(-7, 8) if c != 0])
            if power == 0:
                term = str(coeff)
            elif power == 1:
                term = f"{coeff}*x"
            else:
                term = f"{coeff}*x**{power}"
            terms.append(term)
        quotient_gold = "+".join(terms).replace("+-", "-")
        quotient_bad = quotient_gold.replace("**", "^")
        remainder = str(rng.randint(-15, 15))
        c = rng.randint(-5, 5)
        divisor = f"x-{c}" if c >= 0 else f"x+{abs(c)}"
        question = f"Using synthetic division, divide a polynomial by {divisor}. The quotient is {quotient_bad} and the remainder is requested. Quotient=[ANS] Remainder=[ANS]"
        builder.add("algebra_power_syntax", "caret power syntax where scorer expects Python-style power", question, [quotient_gold, remainder], [quotient_bad, remainder], "Randomized power-syntax case.")

    while needs("direct_proportion_form"):
        z_den = rng.randint(50, 260)
        g_den = rng.randint(12, 48)
        z_inches = str(rng.randint(10, 80))
        real_feet = str(rng.randint(80, 800))
        z_scale = mp.mpf(1) / z_den
        g_scale = mp.mpf(1) / g_den
        inferred_real = mp.mpf(z_inches) / z_scale / 12
        g_inches = mp.mpf(real_feet) * 12 * g_scale
        question = (
            f"A Z scale model train is 1/{z_den}th of the real size and a G scale model train is 1/{g_den}th of the real size. "
            f"The Z scale model is {z_inches} inches long. A real locomotive is {real_feet} feet long. "
            "Give the proportionality model m=[ANS], Z scale k=[ANS], inferred real length in feet=[ANS], G scale k=[ANS], and G model length in inches=[ANS]."
        )
        answer = ["m = k*r", fixed(z_scale, 8), fixed(inferred_real, 3), fixed(g_scale, 7), fmt(g_inches, 15)]
        bad = ["m = k/r", fixed(z_scale, 3), fixed(inferred_real, 0), fixed(g_scale, 3), fixed(g_inches / 12, 3)]
        builder.add("direct_proportion_form", "implicit multiplication and exact fractions where decimals are required", question, answer, bad, "Randomized direct proportion form case.")

    while needs("sequence_canonical_form"):
        power = rng.randint(2, 12)
        answer = [f"(-1)^(n+1)/(n^{power})", f"(n+{power - 1})/(n+{power + 1})"]
        bad = [f"(-1)^n/(n^{power})", f"(n+{power - 1})/(n+{power + 1})"]
        question = (
            f"Find formulas for two sequences. The first starts 1, -1/{2**power}, 1/{3**power}, ... and uses starting index n=1. "
            f"The second has nth term (n+{power - 1})/(n+{power + 1}). First formula=[ANS] Second formula=[ANS]"
        )
        builder.add("sequence_canonical_form", "off-by-one alternating sign convention", question, answer, bad, "Randomized sequence sign-index case.")

    while needs("question_literal_precision"):
        dollars = rng.randint(10, 65)
        cents = rng.choice(["00", "25", "50", "75"])
        base = f"{dollars}.{cents}"
        included = str(rng.choice([150, 200, 250, 300, 450, 500, 750, 900]))
        rate = f"{rng.randint(10, 95) / 100:.2f}"
        answer = [base, "0", included, f"{base}+{rate}*(m-{included})", included]
        bad_base = fmt(mp.mpf(base) + mp.mpf(rate) / 1000, 10)
        bad = [bad_base, "0", included, f"{base}+{rate}*(m-{included})", included]
        question = (
            f"A mobile plan has a base monthly fee of ${base} for the first {included} minutes and charges ${rate} for each additional minute. "
            "Give base fee, initial marginal cost, included minutes, overage formula, and breakpoint. [ANS] [ANS] [ANS] [ANS] [ANS]"
        )
        builder.add("question_literal_precision", "expanded a literal copied from the question", question, answer, bad, "Randomized literal precision case.")


def validate(builder: Builder) -> dict[str, Any]:
    judger = Judger(strict_extract=False)
    gold_failures: list[int] = []
    bad_accepted: list[int] = []
    for item, gold_pred, bad_pred in zip(builder.items, builder.gold_predictions, builder.bad_predictions):
        if not score_item(judger, item, gold_pred["response"]):
            gold_failures.append(int(item["id"]))
        if score_item(judger, item, bad_pred["response"]):
            bad_accepted.append(int(item["id"]))
    if gold_failures or bad_accepted:
        raise RuntimeError(
            "Synthetic validation failed: "
            f"gold_failures={gold_failures[:20]}, bad_accepted={bad_accepted[:20]}"
        )
    return {
        "gold_correct": len(builder.gold_predictions),
        "gold_total": len(builder.gold_predictions),
        "bad_correct": 0,
        "bad_total": len(builder.bad_predictions),
    }


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    builder = Builder(args.start_id)

    add_scientific_notation(builder)
    add_sample_size(builder)
    add_rounding_prompt(builder)
    add_trig_precision(builder)
    add_chi_square(builder)
    add_ttest_lower_tail(builder)
    add_regression_precision(builder)
    add_synthetic_division(builder)
    add_direct_proportion(builder)
    add_sequence_index_form(builder)
    add_piecewise_literals(builder)
    add_extra_cases(builder, args.cases_per_category)

    validation = validate(builder)
    prefix = Path(args.output_prefix)
    dataset_path = prefix.with_suffix(".jsonl")
    gold_path = prefix.with_name(prefix.name + "_gold_responses.jsonl")
    bad_path = prefix.with_name(prefix.name + "_bad_responses.jsonl")
    pairs_path = prefix.with_name(prefix.name + "_finalizer_pairs.jsonl")
    summary_path = prefix.with_name(prefix.name + "_summary.json")

    write_jsonl(dataset_path, builder.items)
    write_jsonl(gold_path, builder.gold_predictions)
    write_jsonl(bad_path, builder.bad_predictions)
    write_jsonl(pairs_path, builder.finalizer_pairs)

    category_counts = Counter(item["synthetic_category"] for item in builder.items)
    target_counts = Counter(item["target_error"] for item in builder.items)
    summary = {
        "seed": SEED,
        "total": len(builder.items),
        "id_range": [builder.items[0]["id"], builder.items[-1]["id"]],
        "files": {
            "dataset": str(dataset_path),
            "gold_responses": str(gold_path),
            "bad_responses": str(bad_path),
            "finalizer_pairs": str(pairs_path),
        },
        "category_counts": dict(sorted(category_counts.items())),
        "target_error_counts": dict(sorted(target_counts.items())),
        "validation": validation,
        "provenance_note": "Template-generated rows; validated separately for zero exact ID/question overlap with data/public.jsonl.",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
