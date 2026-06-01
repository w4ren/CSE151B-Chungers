# Added by Codex: LoRA v2 finalizer SFT builder that avoids public debug slices.

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from math_comp.data import answer_slot_count, is_mcq, read_jsonl, write_jsonl
from math_comp.final_answer import final_format_contract, last_boxed_content
from math_comp.prompts import format_options


SYSTEM = (
    "You are the same Qwen math model. Produce the final answer only. "
    "Use the problem and the previous Qwen reasoning trace. Do not call tools. "
    "Extract and normalize the final answer; do not re-solve unless the trace is unusable. "
    "Obey the requested output schema exactly. Never summarize multiple slots into one value. "
    "Do not include prose outside the final box."
)

DEFAULT_EXCLUDES = [
    "data/public.jsonl",
    "codex_added/job_data/public200_first100_plus_diagnostic100.jsonl",
    "codex_added/job_data/public_second100_freeform_stratified50_v6_eval.jsonl",
    "codex_added/job_data/public_second100_holdout50_nonoverlap_v7_eval.jsonl",
    "codex_added/job_data/public_freeform25_nonoverlap_v8_validation.jsonl",
]

DEFAULT_ALLOWED_TRAIN_SOURCES = ["metamathqa", "mathinstruct", "openmathinstruct2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LoRA v2 finalizer SFT messages from non-public extraction examples."
    )
    parser.add_argument(
        "--synthetic-pairs",
        default="codex_added/job_data/synthetic_precision_common_errors_v2_finalizer_pairs.jsonl",
        help="Synthetic finalizer-pair JSONL.",
    )
    parser.add_argument(
        "--solver-train",
        default="codex_added/solver_sft/data/train.jsonl",
        help="Optional clean external solver-SFT JSONL with problem/reasoning/answer/source.",
    )
    parser.add_argument(
        "--max-clean-train",
        type=int,
        default=900,
        help="Maximum external train-only extraction examples to include.",
    )
    parser.add_argument(
        "--allowed-train-sources",
        nargs="+",
        default=DEFAULT_ALLOWED_TRAIN_SOURCES,
        help="Allowed values from solver-train source field. competition_format is intentionally excluded by default.",
    )
    parser.add_argument(
        "--synthetic-mcq-count",
        type=int,
        default=300,
        help="Number of generated synthetic MCQ extraction examples.",
    )
    parser.add_argument("--seed", type=int, default=1517)
    parser.add_argument("--max-trace-chars", type=int, default=2400)
    parser.add_argument(
        "--exclude-data",
        action="append",
        default=[],
        help="Additional JSONL data files whose exact ids/questions must be excluded.",
    )
    parser.add_argument(
        "--output",
        default="codex_added/data/lora_v2/finalizer_sft_messages.jsonl",
        help="Output SFT JSONL with chat messages.",
    )
    parser.add_argument(
        "--manifest",
        default="codex_added/data/lora_v2/dry_run_manifest.json",
        help="Dry-run/training manifest JSON path.",
    )
    parser.add_argument(
        "--manifest-md",
        default="codex_added/data/lora_v2/dry_run_manifest.md",
        help="Human-readable manifest markdown path.",
    )
    return parser.parse_args()


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def boxed(content: str) -> str:
    return "\\boxed{" + str(content).strip() + "}"


def clean_target_answer(answer: Any) -> str:
    if isinstance(answer, list):
        parts = [clean_target_answer(part).removeprefix("\\boxed{").removesuffix("}") for part in answer]
        return boxed(", ".join(parts))
    text = str(answer).strip()
    inside = last_boxed_content(text)
    if inside:
        return boxed(inside)
    return boxed(text)


def extraction_rules(item: dict[str, Any]) -> str:
    if is_mcq(item):
        return "\n".join(
            [
                "- Choose exactly one option letter from the listed choices.",
                "- If the trace boxed an option value, map that value back to its option letter.",
                "- Output only the option letter, never the option text or numeric value.",
            ]
        )
    slots = answer_slot_count(item)
    return "\n".join(
        [
            f"- The problem has {slots} [ANS] slot(s); output exactly {slots} comma-separated field(s).",
            "- Fill the slots in the same order they appear in the problem.",
            "- If the trace contains a table/list of intermediate requested values, include every requested value.",
            "- Do not keep only the final statistic when earlier table cells are also [ANS] slots.",
            "- Preserve exact expressions and the most precise decimals present in the trace unless the problem explicitly requests rounding.",
        ]
    )


def build_user_prompt(item: dict[str, Any], previous_response: str, max_trace_chars: int) -> str:
    question = str(item["question"])
    if item.get("options"):
        question += "\n\nOptions:\n" + format_options(item["options"])
    trace = str(previous_response)[-max_trace_chars:]
    return (
        "Problem:\n"
        f"{question}\n\n"
        "Required final-answer schema:\n"
        f"{final_format_contract(item)}\n\n"
        "Extraction rules:\n"
        f"{extraction_rules(item)}\n\n"
        "Previous Qwen reasoning trace:\n"
        f"{trace}\n\n"
        "Final answer only:"
    )


def build_record(
    *,
    item_id: int,
    item: dict[str, Any],
    source_response: str,
    target_response: str,
    source_name: str,
    source_kind: str,
    category: str,
    provenance_note: str,
    max_trace_chars: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_prompt(item, source_response, max_trace_chars)},
        {"role": "assistant", "content": target_response},
    ]
    row = {
        "id": item_id,
        "source_name": source_name,
        "source_kind": source_kind,
        "category": category,
        "provenance_note": provenance_note,
        "is_mcq": is_mcq(item),
        "question_hash": stable_hash(normalize_question(item["question"])),
        "target_response": target_response,
        "messages": messages,
    }
    if extra:
        row.update(extra)
    return row


def load_exclusion_sets(paths: Iterable[str]) -> tuple[set[int], set[str], list[dict[str, Any]]]:
    excluded_ids: set[int] = set()
    excluded_questions: set[str] = set()
    sources: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            sources.append({"path": str(path), "exists": False, "rows": 0})
            continue
        rows = read_jsonl(path)
        for row in rows:
            if "id" in row:
                try:
                    excluded_ids.add(int(row["id"]))
                except (TypeError, ValueError):
                    pass
            if row.get("question"):
                excluded_questions.add(normalize_question(row["question"]))
            elif row.get("problem"):
                excluded_questions.add(normalize_question(row["problem"]))
        sources.append({"path": str(path), "exists": True, "rows": len(rows)})
    return excluded_ids, excluded_questions, sources


def should_exclude(
    *,
    item_id: int | None,
    question: str,
    excluded_ids: set[int],
    excluded_questions: set[str],
) -> str | None:
    if item_id is not None and item_id in excluded_ids:
        return "id_overlap"
    if normalize_question(question) in excluded_questions:
        return "exact_question_overlap"
    return None


def load_synthetic_pairs(
    args: argparse.Namespace,
    excluded_ids: set[int],
    excluded_questions: set[str],
    removed: Counter[str],
) -> list[dict[str, Any]]:
    rows = read_jsonl(args.synthetic_pairs)
    records: list[dict[str, Any]] = []
    for row in rows:
        item_id = int(row["id"])
        reason = should_exclude(
            item_id=item_id,
            question=str(row["question"]),
            excluded_ids=excluded_ids,
            excluded_questions=excluded_questions,
        )
        if reason:
            removed[f"synthetic_precision_common_errors_v2:{reason}"] += 1
            continue
        item = {
            "id": item_id,
            "question": row["question"],
            "answer": row["answer"],
            "synthetic_category": row.get("synthetic_category"),
        }
        records.append(
            build_record(
                item_id=item_id,
                item=item,
                source_response=str(row["source_response"]),
                target_response=str(row["target_response"]),
                source_name="synthetic_precision_common_errors_v2_finalizer_pairs",
                source_kind="synthetic",
                category=str(row.get("synthetic_category") or "synthetic_precision"),
                provenance_note=(
                    "targeted synthetic rows; zero exact overlap is checked here, but failure modes were "
                    "designed after public debugging"
                ),
                max_trace_chars=args.max_trace_chars,
                extra={"target_error": row.get("target_error")},
            )
        )
    return records


def synthetic_mcq_item(index: int, letter_index: int, rng: random.Random) -> tuple[dict[str, Any], str]:
    option_count = 5
    answer_value = None
    template = index % 5
    if template == 0:
        a = rng.randint(3, 18)
        b = rng.randint(2, 14)
        answer_value = str(a * b)
        question = f"Which option equals {a} times {b}?"
        distractors = {str(a * b + delta) for delta in [-3, -1, 1, 2, 5]}
        source = (
            f"The computation gives {a}*{b}={answer_value}. That value appears in option "
            f"{chr(65 + letter_index)}, although my raw final box is the value. Final answer: {boxed(answer_value)}"
        )
    elif template == 1:
        n = rng.randint(4, 30)
        answer_value = "True" if n % 2 == 0 else "False"
        question = f"Is {n} an even integer?"
        distractors = {"True", "False", "Cannot be determined", "Both true and false", "Neither"}
        source = (
            f"The statement evaluates to {answer_value}. The option with that value is "
            f"{chr(65 + letter_index)}. I accidentally wrote {boxed(answer_value)} instead of the letter."
        )
    elif template == 2:
        x = rng.randint(2, 12)
        c = rng.randint(2, 9)
        answer_value = str(x)
        question = f"Solve x + {c} = {x + c}. Which option gives x?"
        distractors = {str(x + delta) for delta in [-4, -2, -1, 1, 3, 5]}
        source = (
            f"Subtracting {c} gives x={answer_value}. In the choices this is option "
            f"{chr(65 + letter_index)}. Final answer: {boxed(answer_value)}"
        )
    elif template == 3:
        p = rng.randint(2, 9)
        q = rng.randint(2, 9)
        answer_value = f"{p}/{q}"
        question = f"Which option is the fraction p/q when p={p} and q={q}?"
        distractors = {f"{p + 1}/{q}", f"{p}/{q + 1}", f"{q}/{p}", f"{p*q}", f"{p + q}"}
        source = (
            f"The requested fraction is {answer_value}, matching option {chr(65 + letter_index)}. "
            f"The raw response boxed the expression: {boxed(answer_value)}"
        )
    else:
        n = rng.randint(2, 9)
        answer_value = f"{n}^2"
        question = f"Which option uses power notation for n squared when n={n}?"
        distractors = {f"{n}*2", f"2^{n}", f"{n+n}", f"{n**2}", f"sqrt({n})"}
        source = (
            f"The requested notation is {answer_value}. It is listed as option {chr(65 + letter_index)}. "
            f"Final answer should be the letter, not {boxed(answer_value)}."
        )

    distractor_list = [value for value in sorted(distractors) if value != answer_value]
    while len(distractor_list) < option_count - 1:
        distractor_list.append(f"none-{index}-{len(distractor_list)}")
    options = distractor_list[: option_count - 1]
    options.insert(letter_index, answer_value)
    item = {
        "id": 950000 + index,
        "question": question,
        "options": options,
        "answer": chr(65 + letter_index),
    }
    return item, source


def build_synthetic_mcq(
    args: argparse.Namespace,
    excluded_ids: set[int],
    excluded_questions: set[str],
    removed: Counter[str],
) -> list[dict[str, Any]]:
    rng = random.Random(args.seed + 101)
    records: list[dict[str, Any]] = []
    for index in range(args.synthetic_mcq_count):
        letter_index = index % 5
        item, source_response = synthetic_mcq_item(index, letter_index, rng)
        reason = should_exclude(
            item_id=int(item["id"]),
            question=str(item["question"]),
            excluded_ids=excluded_ids,
            excluded_questions=excluded_questions,
        )
        if reason:
            removed[f"synthetic_mcq_option_letter:{reason}"] += 1
            continue
        records.append(
            build_record(
                item_id=int(item["id"]),
                item=item,
                source_response=source_response,
                target_response=boxed(item["answer"]),
                source_name="synthetic_mcq_option_letter",
                source_kind="synthetic",
                category="mcq_option_letter",
                provenance_note="template-generated synthetic MCQ rows with exact public-question exclusion",
                max_trace_chars=args.max_trace_chars,
            )
        )
    return records


def load_clean_train_examples(
    args: argparse.Namespace,
    excluded_questions: set[str],
    removed: Counter[str],
) -> list[dict[str, Any]]:
    path = Path(args.solver_train)
    if args.max_clean_train <= 0 or not path.exists():
        return []

    rng = random.Random(args.seed + 202)
    allowed = set(args.allowed_train_sources)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        source = str(row.get("source") or "unknown")
        if source not in allowed:
            removed[f"solver_train:{source}:source_not_allowed"] += 1
            continue
        problem = str(row.get("problem") or row.get("question") or "").strip()
        answer = row.get("answer")
        reasoning = str(row.get("reasoning") or "").strip()
        if not problem or not answer:
            removed[f"solver_train:{source}:missing_problem_or_answer"] += 1
            continue
        reason = should_exclude(
            item_id=None,
            question=problem,
            excluded_ids=set(),
            excluded_questions=excluded_questions,
        )
        if reason:
            removed[f"solver_train:{source}:{reason}"] += 1
            continue
        if len(problem) > 1500:
            removed[f"solver_train:{source}:problem_too_long"] += 1
            continue
        by_source[source].append(row)

    selected: list[dict[str, Any]] = []
    per_source = max(1, args.max_clean_train // max(1, len(allowed)))
    for source in sorted(allowed):
        candidates = by_source.get(source, [])
        rng.shuffle(candidates)
        selected.extend(candidates[:per_source])

    if len(selected) < args.max_clean_train:
        already = {id(row) for row in selected}
        remaining = [row for rows in by_source.values() for row in rows if id(row) not in already]
        rng.shuffle(remaining)
        selected.extend(remaining[: args.max_clean_train - len(selected)])
    else:
        selected = selected[: args.max_clean_train]

    records: list[dict[str, Any]] = []
    for offset, row in enumerate(selected):
        source = str(row.get("source") or "unknown")
        problem = str(row.get("problem") or row.get("question") or "").strip()
        answer = clean_target_answer(row.get("answer"))
        reasoning = str(row.get("reasoning") or "").strip()
        source_response = f"{reasoning}\n\nFinal answer: {answer}".strip()
        item = {"id": 960000 + offset, "question": problem, "answer": answer}
        records.append(
            build_record(
                item_id=960000 + offset,
                item=item,
                source_response=source_response,
                target_response=answer,
                source_name=f"solver_sft_train:{source}",
                source_kind="train",
                category=f"external_solution_trace:{source}",
                provenance_note="external train-only source after exact public-question exclusion",
                max_trace_chars=args.max_trace_chars,
            )
        )
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(row["source_name"] for row in records)
    by_kind = Counter(row["source_kind"] for row in records)
    by_category = Counter(row["category"] for row in records)
    by_risk = Counter(row["provenance_note"] for row in records)
    mcq = sum(bool(row["is_mcq"]) for row in records)
    return {
        "total_examples": len(records),
        "mcq_examples": mcq,
        "free_form_examples": len(records) - mcq,
        "by_source": dict(sorted(by_source.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_provenance_note": dict(sorted(by_risk.items())),
    }


def write_manifest_md(path: Path, manifest: dict[str, Any]) -> None:
    lines = ["# LoRA v2 Finalizer SFT Dry-Run Manifest", ""]
    lines.append(f"- Output SFT: `{manifest['outputs']['sft_jsonl']}`")
    lines.append(f"- Planned adapter: `{manifest['outputs']['planned_adapter_dir']}`")
    lines.append(f"- Total examples: {manifest['summary']['total_examples']}")
    lines.append(f"- MCQ examples: {manifest['summary']['mcq_examples']}")
    lines.append(f"- Free-form examples: {manifest['summary']['free_form_examples']}")
    lines.append("")
    lines.append("## Sources")
    lines.append("| Source | Kind | Examples | Provenance note |")
    lines.append("|---|---|---:|---|")
    source_meta = manifest["source_metadata"]
    for source, count in manifest["summary"]["by_source"].items():
        meta = source_meta.get(source, {})
        lines.append(f"| `{source}` | {meta.get('kind', '')} | {count} | {meta.get('provenance_note', '')} |")
    lines.append("")
    lines.append("## Categories")
    lines.append("| Category | Examples |")
    lines.append("|---|---:|")
    for category, count in manifest["summary"]["by_category"].items():
        lines.append(f"| `{category}` | {count} |")
    lines.append("")
    lines.append("## Exclusion Checks")
    for source in manifest["exclusion_sources"]:
        state = "found" if source["exists"] else "missing"
        lines.append(f"- `{source['path']}`: {state}, rows={source['rows']}")
    lines.append("")
    lines.append("## Removed Rows")
    if manifest["removed_rows"]:
        for reason, count in manifest["removed_rows"].items():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Notes")
    for note in manifest["notes"]:
        lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    exclude_paths = [*DEFAULT_EXCLUDES, *args.exclude_data]
    excluded_ids, excluded_questions, exclusion_sources = load_exclusion_sets(exclude_paths)
    removed: Counter[str] = Counter()

    records: list[dict[str, Any]] = []
    records.extend(load_synthetic_pairs(args, excluded_ids, excluded_questions, removed))
    records.extend(build_synthetic_mcq(args, excluded_ids, excluded_questions, removed))
    records.extend(load_clean_train_examples(args, excluded_questions, removed))

    rng = random.Random(args.seed)
    rng.shuffle(records)

    output_path = Path(args.output)
    write_jsonl(output_path, records)

    source_metadata: dict[str, dict[str, Any]] = {}
    for row in records:
        source_metadata.setdefault(
            row["source_name"],
            {"kind": row["source_kind"], "provenance_note": row["provenance_note"]},
        )

    manifest = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "seed": args.seed,
        "max_trace_chars": args.max_trace_chars,
        "inputs": {
            "synthetic_pairs": args.synthetic_pairs,
            "solver_train": args.solver_train if Path(args.solver_train).exists() else None,
            "raw_train_outputs": None,
        },
        "outputs": {
            "sft_jsonl": str(output_path),
            "manifest_json": args.manifest,
            "manifest_md": args.manifest_md,
            "planned_adapter_dir": "codex_added/models/qwen3_answer_format_lora_v2",
        },
        "summary": summarize_records(records),
        "source_metadata": source_metadata,
        "exclusion_sources": exclusion_sources,
        "removed_rows": dict(sorted(removed.items())),
        "notes": [
            "Public200, freeform50-style public slices, and data/public.jsonl are exclusion sources only, not training sources.",
            "Synthetic v2 has exact ID/question overlap excluded here, but it remains targeted to public-audited failure modes.",
            "External solver train rows are used as answer-extraction traces, not question-only solver examples.",
            "No raw train-output JSONL was found or provided for this build.",
            "MCQ synthetic rows explicitly train value-to-option-letter extraction, targeting boxed-value MCQ failures.",
        ],
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest_md(Path(args.manifest_md), manifest)

    print(f"Wrote {len(records)} LoRA v2 SFT examples to {output_path}")
    print(f"Wrote manifest to {manifest_path}")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
