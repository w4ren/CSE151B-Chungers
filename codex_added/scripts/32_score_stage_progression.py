# Added by Codex: score raw/finalized/repaired stages side-by-side.

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from judger import Judger
from math_comp.data import answer_slot_count, has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.final_answer import final_answer_diagnostics
from math_comp.scoring import score_item, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score multiple pipeline stages and write progression summaries.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--stage", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--scored-dir", default=None)
    return parser.parse_args()


def group_for(item: dict[str, Any]) -> str:
    if item.get("diagnostic_group"):
        return str(item["diagnostic_group"])
    if is_mcq(item):
        return "mcq"
    if answer_slot_count(item) >= 2:
        return "multi_slot_freeform"
    return "freeform_single_slot"


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    mismatches = sum(bool(row.get("slot_mismatch")) for row in rows)
    format_ok = sum(bool(row.get("format_ok")) for row in rows)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "format_ok": format_ok,
        "slot_mismatches": mismatches,
    }


def score_stage(
    label: str,
    path: str,
    data: list[dict[str, Any]],
    data_by_id: dict[int, dict[str, Any]],
    judger: Judger,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_by_id = index_by_id(read_jsonl(path))
    scored: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        item_id = int(item["id"])
        pred = pred_by_id.get(item_id, {"id": item_id, "response": ""})
        response = str(pred.get("response", ""))
        diagnostics = final_answer_diagnostics(item, response)
        row = {
            **pred,
            "id": item_id,
            "stage": label,
            "is_mcq": is_mcq(item),
            "diagnostic_group": group_for(item),
            "expected_slots": diagnostics["expected_slots"],
            "parsed_slots": diagnostics["parsed_slots"],
            "format_ok": diagnostics["format_ok"],
            "slot_mismatch": (not is_mcq(item)) and diagnostics["parsed_slots"] != diagnostics["expected_slots"],
        }
        if has_gold(item):
            row["gold"] = item["answer"]
            row["correct"] = score_item(judger, item, response)
        scored.append(row)
        grouped[row["diagnostic_group"]].append(row)

    summary = summarize_results(scored)
    summary["groups"] = {group: summarize_group(rows) for group, rows in sorted(grouped.items())}
    summary["format_ok"] = sum(bool(row["format_ok"]) for row in scored)
    summary["slot_mismatches"] = sum(bool(row["slot_mismatch"]) for row in scored)
    return summary, scored


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    data_by_id = index_by_id(data)
    judger = Judger(strict_extract=False)
    summaries: dict[str, Any] = {}

    scored_dir = Path(args.scored_dir) if args.scored_dir else None
    if scored_dir:
        scored_dir.mkdir(parents=True, exist_ok=True)

    for label, path in args.stage:
        summary, scored = score_stage(label, path, data, data_by_id, judger)
        summaries[label] = summary
        if scored_dir:
            write_jsonl(scored_dir / f"{label}_scored.jsonl", scored)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = ["# Stage Progression", ""]
    lines.append("| Stage | Overall | MCQ | Free-form | Format OK | Slot mismatches |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, summary in summaries.items():
        overall = summary["overall"]
        mcq = summary["mcq"]
        free = summary["free_form"]
        lines.append(
            "| {label} | {oc}/{ot} | {mc}/{mt} | {fc}/{ft} | {fmt}/{ot} | {slot} |".format(
                label=label,
                oc=overall["correct"],
                ot=overall["total"],
                mc=mcq["correct"],
                mt=mcq["total"],
                fc=free["correct"],
                ft=free["total"],
                fmt=summary["format_ok"],
                slot=summary["slot_mismatches"],
            )
        )
    lines.append("")
    lines.append("## Diagnostic Groups")
    for label, summary in summaries.items():
        lines.append("")
        lines.append(f"### {label}")
        lines.append("| Group | Correct | Accuracy | Format OK | Slot mismatches |")
        lines.append("|---|---:|---:|---:|---:|")
        for group, row in summary["groups"].items():
            lines.append(
                f"| {group} | {row['correct']}/{row['total']} | {row['accuracy']:.4f} | "
                f"{row['format_ok']} | {row['slot_mismatches']} |"
            )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote stage summaries to {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
