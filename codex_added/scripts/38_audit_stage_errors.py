# Added by Codex: categorize and plot right/wrong outcomes across pipeline stages.

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from judger import Judger
from math_comp.data import has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.scoring import extract_answer_key, score_item


def load_verifier_module() -> Any:
    path = Path(__file__).with_name("31_verify_repair_with_qwen.py")
    spec = importlib.util.spec_from_file_location("strict_verify_repair", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load verifier module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit right/wrong and error categories for pipeline stages.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--stage", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def short_text(value: Any, limit: int = 220) -> str:
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def category_for(item: dict[str, Any], response: str, correct: bool, verification: dict[str, Any]) -> str:
    if correct:
        return "correct"
    error_type = str(verification.get("error_type", "none"))
    if not response.strip():
        return "missing_prediction"
    if error_type in {"missing_box", "multiple_boxes", "malformed_answer"}:
        return f"format:{error_type}"
    if error_type in {"wrong_slot_count", "truncated_or_incomplete"}:
        return f"slot_or_completion:{error_type}"
    if error_type == "precision_mismatch":
        return "form:precision_mismatch"
    if error_type == "exact_form_mismatch":
        return "form:exact_form_mismatch"
    if error_type == "mcq_invalid_letter":
        return "mcq:invalid_letter"
    if is_mcq(item):
        return "mcq:wrong_option"
    return "actual_math_error"


def score_stage(
    label: str,
    path: str,
    data: list[dict[str, Any]],
    verifier: Any,
    judger: Judger,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_by_id = index_by_id(read_jsonl(path))
    rows: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    wrong_categories: Counter[str] = Counter()
    strict_errors: Counter[str] = Counter()
    correct_count = 0
    strict_valid_count = 0

    for item in data:
        item_id = int(item["id"])
        pred = pred_by_id.get(item_id, {"id": item_id, "response": ""})
        response = str(pred.get("response", ""))
        verification = verifier.strict_verify_final_answer(item, response)
        correct = bool(score_item(judger, item, response)) if has_gold(item) else False
        category = category_for(item, response, correct, verification)
        categories[category] += 1
        if not correct:
            wrong_categories[category] += 1
        strict_errors[str(verification.get("error_type", "none"))] += 1
        correct_count += int(correct)
        strict_valid_count += int(bool(verification.get("valid")))
        rows.append(
            {
                "stage": label,
                "id": item_id,
                "is_mcq": is_mcq(item),
                "correct": correct,
                "wrong": not correct,
                "strict_valid": bool(verification.get("valid")),
                "error_type": str(verification.get("error_type", "none")),
                "category": category,
                "expected_slots": int(verification.get("expected_slots", 0)),
                "predicted_slots": int(verification.get("predicted_slots", 0)),
                "answer_key": extract_answer_key(item, response, strict=True),
                "gold": item.get("answer"),
                "response": response,
                "question_preview": short_text(item.get("question", "")),
            }
        )

    total = len(data)
    summary = {
        "correct": correct_count,
        "wrong": total - correct_count,
        "total": total,
        "accuracy": correct_count / total if total else 0.0,
        "strict_valid": strict_valid_count,
        "strict_invalid": total - strict_valid_count,
        "categories": dict(sorted(categories.items())),
        "wrong_categories": dict(sorted(wrong_categories.items())),
        "strict_errors": dict(sorted(strict_errors.items())),
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "id",
        "is_mcq",
        "correct",
        "wrong",
        "strict_valid",
        "error_type",
        "category",
        "expected_slots",
        "predicted_slots",
        "answer_key",
        "gold",
        "question_preview",
        "response",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def svg_bar_chart(summary: dict[str, Any], path: Path) -> None:
    labels = list(summary)
    width = 920
    height = 320
    margin_left = 100
    margin_bottom = 70
    plot_width = width - margin_left - 40
    plot_height = height - 60 - margin_bottom
    max_total = max((summary[label]["total"] for label in labels), default=1)
    bar_width = max(42, min(110, plot_width // max(1, len(labels)) - 34))
    gap = (plot_width - bar_width * len(labels)) / max(1, len(labels))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-family="Arial" font-size="20" font-weight="700">Correct vs wrong by stage</text>',
        '<text x="760" y="32" font-family="Arial" font-size="13" fill="#14804a">correct</text>',
        '<rect x="738" y="22" width="14" height="14" fill="#2ca25f"/>',
        '<text x="842" y="32" font-family="Arial" font-size="13" fill="#a13b2a">wrong</text>',
        '<rect x="820" y="22" width="14" height="14" fill="#de6b48"/>',
    ]
    axis_y = height - margin_bottom
    parts.append(f'<line x1="{margin_left}" y1="{axis_y}" x2="{width - 30}" y2="{axis_y}" stroke="#333"/>')
    for idx, label in enumerate(labels):
        x = margin_left + gap / 2 + idx * (bar_width + gap)
        correct = summary[label]["correct"]
        wrong = summary[label]["wrong"]
        correct_h = plot_height * correct / max_total
        wrong_h = plot_height * wrong / max_total
        y_wrong = axis_y - wrong_h
        y_correct = y_wrong - correct_h
        parts.append(f'<rect x="{x:.1f}" y="{y_correct:.1f}" width="{bar_width}" height="{correct_h:.1f}" fill="#2ca25f"/>')
        parts.append(f'<rect x="{x:.1f}" y="{y_wrong:.1f}" width="{bar_width}" height="{wrong_h:.1f}" fill="#de6b48"/>')
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y_correct - 8:.1f}" text-anchor="middle" '
            f'font-family="Arial" font-size="13">{correct}/{correct + wrong}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{axis_y + 22}" text-anchor="middle" '
            f'font-family="Arial" font-size="12">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def svg_category_chart(summary: dict[str, Any], path: Path) -> None:
    labels = list(summary)
    category_names = sorted({cat for row in summary.values() for cat in row["wrong_categories"]})
    palette = [
        "#6c8ebf",
        "#b85450",
        "#d6b656",
        "#82b366",
        "#9673a6",
        "#d79b00",
        "#76a5af",
        "#cc4125",
        "#3d85c6",
        "#674ea7",
    ]
    colors = {cat: palette[index % len(palette)] for index, cat in enumerate(category_names)}
    width = 1040
    height = max(360, 210 + 22 * len(category_names))
    margin_left = 110
    margin_bottom = 80
    plot_width = 640
    plot_height = 190
    max_wrong = max((row["wrong"] for row in summary.values()), default=1) or 1
    bar_width = max(42, min(110, plot_width // max(1, len(labels)) - 34))
    gap = (plot_width - bar_width * len(labels)) / max(1, len(labels))
    axis_y = 250
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-family="Arial" font-size="20" font-weight="700">Wrong-answer categories by stage</text>',
        f'<line x1="{margin_left}" y1="{axis_y}" x2="{margin_left + plot_width}" y2="{axis_y}" stroke="#333"/>',
    ]
    for idx, label in enumerate(labels):
        x = margin_left + gap / 2 + idx * (bar_width + gap)
        y = axis_y
        for cat in category_names:
            count = summary[label]["wrong_categories"].get(cat, 0)
            h = plot_height * count / max_wrong
            y -= h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{h:.1f}" '
                f'fill="{colors[cat]}"><title>{html.escape(cat)}: {count}</title></rect>'
            )
        wrong = summary[label]["wrong"]
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
            f'font-family="Arial" font-size="13">{wrong}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{axis_y + 22}" text-anchor="middle" '
            f'font-family="Arial" font-size="12">{html.escape(label)}</text>'
        )

    legend_x = 800
    legend_y = 64
    for index, cat in enumerate(category_names):
        y = legend_y + index * 22
        parts.append(f'<rect x="{legend_x}" y="{y - 11}" width="14" height="14" fill="{colors[cat]}"/>')
        parts.append(
            f'<text x="{legend_x + 22}" y="{y}" font-family="Arial" font-size="12">{html.escape(cat)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = ["# Stage Error Audit", ""]
    lines.append("| Stage | Correct | Wrong | Accuracy | Strict valid | Strict invalid |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, row in summary.items():
        lines.append(
            f"| {label} | {row['correct']} | {row['wrong']} | {row['accuracy']:.4f} | "
            f"{row['strict_valid']} | {row['strict_invalid']} |"
        )
    lines.append("")
    lines.append("## Wrong Categories")
    for label, row in summary.items():
        lines.append("")
        lines.append(f"### {label}")
        if not row["wrong_categories"]:
            lines.append("No wrong rows.")
            continue
        lines.append("| Category | Count |")
        lines.append("|---|---:|")
        for category, count in sorted(row["wrong_categories"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| {category} | {count} |")

    lines.append("")
    lines.append("## Wrong Row Index")
    lines.append("| Stage | ID | Category | Error type | Gold | Pred key | Question preview |")
    lines.append("|---|---:|---|---|---|---|---|")
    for row in rows:
        if row["correct"]:
            continue
        lines.append(
            "| {stage} | {id} | {category} | {error_type} | {gold} | {answer_key} | {question} |".format(
                stage=row["stage"],
                id=row["id"],
                category=row["category"],
                error_type=row["error_type"],
                gold=html.escape(short_text(row.get("gold", ""), 80)),
                answer_key=html.escape(short_text(row.get("answer_key", ""), 80)),
                question=html.escape(short_text(row.get("question_preview", ""), 120)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verifier = load_verifier_module()
    data = read_jsonl(args.data)
    judger = Judger(strict_extract=False)

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for label, path in args.stage:
        summary, rows = score_stage(label, path, data, verifier, judger)
        summaries[label] = summary
        all_rows.extend(rows)

    write_jsonl(out_dir / "per_stage_audit.jsonl", all_rows)
    write_csv(out_dir / "per_stage_audit.csv", all_rows)
    (out_dir / "error_audit_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "error_audit.md", summaries, all_rows)
    svg_bar_chart(summaries, out_dir / "correct_wrong_by_stage.svg")
    svg_category_chart(summaries, out_dir / "wrong_categories_by_stage.svg")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
