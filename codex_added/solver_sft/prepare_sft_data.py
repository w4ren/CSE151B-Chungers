# Added by Codex: build curated solver-SFT data for Qwen math reasoning.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from math_comp.data import is_mcq
from math_comp.prompts import format_options


SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step-by-step and put "
    "the final answer inside \\boxed{}."
)

BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
FINAL_LINE_RE = re.compile(
    r"(?:final\s+answer|answer|the\s+answer|therefore|thus)\s*(?:is|=|:)?\s*(.+)",
    re.IGNORECASE,
)
NOISY_RE = re.compile(
    r"(```|\\begin\{code\}|python|sympy|sage|mathematica|"
    r"write a program|run code|execute code|use code|simulation)",
    re.IGNORECASE,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_csv_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML to use --config.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def dedupe_key(problem: str) -> str:
    normalized = normalize_space(problem).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def ensure_boxed(answer: Any) -> str:
    if isinstance(answer, list):
        answer = ", ".join(str(part).strip() for part in answer)
    text = normalize_space(answer).strip("$")
    boxed = BOXED_RE.findall(text)
    if boxed:
        return f"\\boxed{{{boxed[-1].strip()}}}"
    text = text.strip().rstrip(".")
    if text.startswith("\\boxed{") and text.endswith("}"):
        return text
    return f"\\boxed{{{text}}}" if text else ""


def answer_content(answer: str) -> str:
    boxed = BOXED_RE.findall(answer)
    return boxed[-1].strip() if boxed else str(answer).strip()


def extract_answer_from_text(text: str) -> str:
    boxed = BOXED_RE.findall(text)
    if boxed:
        return ensure_boxed(boxed[-1])

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for line in reversed(lines[-8:]):
        if "####" in line:
            tail = line.rsplit("####", 1)[-1]
            return ensure_boxed(tail)
        match = FINAL_LINE_RE.search(line)
        if match:
            candidate = match.group(1).strip().strip("$").rstrip(".")
            if 0 < len(candidate) <= 240:
                return ensure_boxed(candidate)

    if lines:
        last = lines[-1].strip().strip("$").rstrip(".")
        if 0 < len(last) <= 120 and not last.endswith("?"):
            return ensure_boxed(last)
    return ""


def strip_final_answer_from_reasoning(solution: str, boxed_answer: str) -> str:
    text = str(solution or "").strip()
    if not text:
        return ""
    matches = list(BOXED_RE.finditer(text))
    if matches:
        text = text[: matches[-1].start()].strip()
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and FINAL_LINE_RE.search(lines[-1]):
        lines.pop()
    text = "\n".join(lines).strip()
    content = answer_content(boxed_answer)
    if content and text.endswith(content):
        text = text[: -len(content)].strip()
    return text


def too_noisy(problem: str, reasoning: str) -> bool:
    joined = f"{problem}\n{reasoning}"
    return bool(NOISY_RE.search(joined))


def maybe_truncate(text: str, max_chars: int, truncate: bool) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if not truncate:
        return ""
    return text[:max_chars].rsplit("\n", 1)[0].strip()


def valid_sample(
    sample: dict[str, str],
    *,
    min_reasoning_chars: int,
    max_problem_chars: int,
    max_answer_chars: int,
) -> bool:
    problem = sample.get("problem", "").strip()
    reasoning = sample.get("reasoning", "").strip()
    answer = sample.get("answer", "").strip()
    if len(problem) < 8 or len(problem) > max_problem_chars:
        return False
    if len(reasoning) < min_reasoning_chars:
        return False
    if not answer or len(answer) > max_answer_chars:
        return False
    if not BOXED_RE.search(answer):
        return False
    if "[ANS]" in answer:
        return False
    if too_noisy(problem, reasoning):
        return False
    return True


def first_nonempty(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", []):
            return value
    return ""


def build_problem_from_row(row: dict[str, Any]) -> str:
    problem = first_nonempty(
        row,
        [
            "problem",
            "question",
            "query",
            "instruction",
            "prompt",
            "input",
            "text",
        ],
    )
    if isinstance(problem, dict):
        problem = json.dumps(problem, ensure_ascii=False)
    problem = str(problem or "").strip()
    extra_input = row.get("input")
    if extra_input and str(extra_input).strip() and str(extra_input).strip() not in problem:
        problem = f"{problem}\n\n{extra_input}"
    return problem.strip()


def build_solution_from_row(row: dict[str, Any]) -> str:
    solution = first_nonempty(
        row,
        [
            "reasoning",
            "solution",
            "response",
            "output",
            "generated_solution",
            "rationale",
            "completion",
        ],
    )
    if isinstance(solution, dict):
        solution = json.dumps(solution, ensure_ascii=False)
    return str(solution or "").strip()


def build_answer_from_row(row: dict[str, Any], solution: str) -> str:
    answer = first_nonempty(
        row,
        [
            "answer",
            "final_answer",
            "expected_answer",
            "ground_truth",
            "target",
            "label",
        ],
    )
    if answer not in (None, "", []):
        return ensure_boxed(answer)
    return extract_answer_from_text(solution)


def normalize_external_row(
    row: dict[str, Any],
    source: str,
    *,
    min_reasoning_chars: int,
    max_problem_chars: int,
    max_answer_chars: int,
    max_reasoning_chars: int,
    truncate_long_reasoning: bool,
) -> dict[str, str] | None:
    problem = build_problem_from_row(row)
    solution = build_solution_from_row(row)
    answer = build_answer_from_row(row, solution)
    if not problem or not solution or not answer:
        return None
    reasoning = strip_final_answer_from_reasoning(solution, answer) or solution
    reasoning = maybe_truncate(reasoning, max_reasoning_chars, truncate_long_reasoning)
    sample = {
        "problem": problem,
        "reasoning": reasoning,
        "answer": answer,
        "source": source,
    }
    if not valid_sample(
        sample,
        min_reasoning_chars=min_reasoning_chars,
        max_problem_chars=max_problem_chars,
        max_answer_chars=max_answer_chars,
    ):
        return None
    return sample


def normalize_self_trace(
    row: dict[str, Any],
    *,
    min_reasoning_chars: int,
    max_problem_chars: int,
    max_answer_chars: int,
    max_reasoning_chars: int,
    truncate_long_reasoning: bool,
) -> dict[str, str] | None:
    if row.get("judge_correct") is not True:
        return None
    problem = str(row.get("problem") or row.get("question") or "").strip()
    reasoning = str(row.get("reasoning") or row.get("response") or "").strip()
    answer = ensure_boxed(row.get("answer") or extract_answer_from_text(reasoning))
    reasoning = strip_final_answer_from_reasoning(reasoning, answer) or reasoning
    reasoning = maybe_truncate(reasoning, max_reasoning_chars, truncate_long_reasoning)
    sample = {
        "problem": problem,
        "reasoning": reasoning,
        "answer": answer,
        "source": str(row.get("source") or "qwen_self_sample"),
    }
    if not valid_sample(
        sample,
        min_reasoning_chars=min_reasoning_chars,
        max_problem_chars=max_problem_chars,
        max_answer_chars=max_answer_chars,
    ):
        return None
    return sample


def competition_problem(row: dict[str, Any]) -> str:
    problem = str(row.get("question") or row.get("problem") or "").strip()
    options = row.get("options") or []
    if options:
        problem += "\n\nOptions:\n" + format_options([str(option) for option in options])
    return problem


def normalize_competition_row(
    row: dict[str, Any],
    *,
    min_reasoning_chars: int,
    max_problem_chars: int,
    max_answer_chars: int,
) -> dict[str, str] | None:
    if "answer" not in row or row.get("answer") in (None, "", []):
        return None
    answer = ensure_boxed(row["answer"])
    if is_mcq(row):
        reasoning = (
            "Identify the requested quantity, compare it with the answer choices, "
            f"and select the option letter {answer_content(answer)}."
        )
    else:
        slots = str(row.get("question", "")).count("[ANS]") or 1
        reasoning = (
            f"Solve the problem and fill the {slots} requested [ANS] slot(s) "
            "in order. Keep the final response as one boxed answer."
        )
    sample = {
        "problem": competition_problem(row),
        "reasoning": reasoning,
        "answer": answer,
        "source": "competition_format",
    }
    if not valid_sample(
        sample,
        min_reasoning_chars=min(20, min_reasoning_chars),
        max_problem_chars=max_problem_chars,
        max_answer_chars=max_answer_chars,
    ):
        return None
    return sample


def iter_hf_rows(
    dataset_name: str,
    split: str,
    *,
    streaming: bool,
    seed: int,
    cache_dir: str | None,
) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to load HuggingFace datasets.") from exc

    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    dataset = load_dataset(dataset_name, **kwargs)
    if streaming:
        dataset = dataset.shuffle(seed=seed, buffer_size=20_000)
    else:
        dataset = dataset.shuffle(seed=seed)
    for row in dataset:
        yield dict(row)


def take_hf_samples(
    *,
    dataset_name: str,
    split: str,
    source: str,
    target_count: int,
    seen: set[str],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    if target_count <= 0:
        return []
    rows: list[dict[str, str]] = []
    scanned = 0
    max_scan = max(target_count * args.max_scan_multiplier, target_count)
    for row in iter_hf_rows(
        dataset_name,
        split,
        streaming=args.streaming,
        seed=args.seed + len(seen),
        cache_dir=args.hf_cache_dir,
    ):
        scanned += 1
        sample = normalize_external_row(
            row,
            source,
            min_reasoning_chars=args.min_reasoning_chars,
            max_problem_chars=args.max_problem_chars,
            max_answer_chars=args.max_answer_chars,
            max_reasoning_chars=args.max_reasoning_chars,
            truncate_long_reasoning=args.truncate_long_reasoning,
        )
        if sample:
            key = dedupe_key(sample["problem"])
            if key not in seen:
                seen.add(key)
                rows.append(sample)
                if len(rows) >= target_count:
                    break
        if scanned >= max_scan:
            break
    print(f"{source}: kept {len(rows)} / scanned {scanned}")
    return rows


def take_local_self_samples(paths: list[str], target_count: int, seen: set[str], args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if target_count <= 0:
        return rows
    for path in paths:
        if not path:
            continue
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"warning: missing self-trace file: {path}")
            continue
        for row in read_jsonl(path_obj):
            sample = normalize_self_trace(
                row,
                min_reasoning_chars=args.min_reasoning_chars,
                max_problem_chars=args.max_problem_chars,
                max_answer_chars=args.max_answer_chars,
                max_reasoning_chars=args.max_reasoning_chars,
                truncate_long_reasoning=args.truncate_long_reasoning,
            )
            if not sample:
                continue
            key = dedupe_key(sample["problem"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(sample)
            if len(rows) >= target_count:
                print(f"self_traces: kept {len(rows)}")
                return rows
    print(f"self_traces: kept {len(rows)}")
    return rows


def take_competition_samples(paths: list[str], target_count: int, seen: set[str], args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if target_count <= 0:
        return rows
    for path in paths:
        if not path:
            continue
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"warning: missing competition file: {path}")
            continue
        raw_rows = read_jsonl(path_obj) if path_obj.suffix.lower() in {".jsonl", ".json"} else read_csv_records(path_obj)
        for row in raw_rows:
            sample = normalize_competition_row(
                row,
                min_reasoning_chars=args.min_reasoning_chars,
                max_problem_chars=args.max_problem_chars,
                max_answer_chars=args.max_answer_chars,
            )
            if not sample:
                continue
            key = dedupe_key(sample["problem"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(sample)
            if len(rows) >= target_count:
                print(f"competition_format: kept {len(rows)}")
                return rows
    print(f"competition_format: kept {len(rows)}")
    return rows


def split_train_eval(samples: list[dict[str, str]], eval_size: int, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    eval_n = min(max(eval_size, 0), max(len(shuffled) - 1, 0))
    eval_rows = shuffled[:eval_n]
    train_rows = shuffled[eval_n:]
    return train_rows, eval_rows


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare curated Qwen solver-SFT data.")
    parser.add_argument("--config", default=None, help="Optional YAML config.")
    parser.add_argument("--output-dir", default=defaults.get("output_dir", "codex_added/solver_sft/data"))
    parser.add_argument("--train-file", default=defaults.get("train_file", None))
    parser.add_argument("--eval-file", default=defaults.get("eval_file", None))
    parser.add_argument("--metamath-count", type=int, default=int(defaults.get("metamath_count", 20_000)))
    parser.add_argument("--mathinstruct-count", type=int, default=int(defaults.get("mathinstruct_count", 7_500)))
    parser.add_argument("--openmath-count", type=int, default=int(defaults.get("openmath_count", 7_500)))
    parser.add_argument("--self-count", type=int, default=int(defaults.get("self_count", 10_000)))
    parser.add_argument("--competition-count", type=int, default=int(defaults.get("competition_count", 5_000)))
    parser.add_argument("--metamath-dataset", default=defaults.get("metamath_dataset", "meta-math/MetaMathQA"))
    parser.add_argument("--mathinstruct-dataset", default=defaults.get("mathinstruct_dataset", "TIGER-Lab/MathInstruct"))
    parser.add_argument("--openmath-dataset", default=defaults.get("openmath_dataset", "nvidia/OpenMathInstruct-2"))
    parser.add_argument("--metamath-split", default=defaults.get("metamath_split", "train"))
    parser.add_argument("--mathinstruct-split", default=defaults.get("mathinstruct_split", "train"))
    parser.add_argument("--openmath-split", default=defaults.get("openmath_split", "train"))
    parser.add_argument("--self-traces", nargs="*", default=defaults.get("self_traces", []))
    parser.add_argument("--competition-jsonl", nargs="*", default=defaults.get("competition_jsonl", ["data/public.jsonl"]))
    parser.add_argument("--eval-size", type=int, default=int(defaults.get("eval_size", 1000)))
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 151)))
    parser.add_argument("--hf-cache-dir", default=defaults.get("hf_cache_dir", os.environ.get("HF_DATASETS_CACHE")))
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=bool(defaults.get("streaming", True)))
    parser.add_argument("--max-scan-multiplier", type=int, default=int(defaults.get("max_scan_multiplier", 25)))
    parser.add_argument("--min-reasoning-chars", type=int, default=int(defaults.get("min_reasoning_chars", 80)))
    parser.add_argument("--max-problem-chars", type=int, default=int(defaults.get("max_problem_chars", 12_000)))
    parser.add_argument("--max-answer-chars", type=int, default=int(defaults.get("max_answer_chars", 400)))
    parser.add_argument("--max-reasoning-chars", type=int, default=int(defaults.get("max_reasoning_chars", 32_000)))
    parser.add_argument(
        "--truncate-long-reasoning",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("truncate_long_reasoning", True)),
    )
    return parser


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, remaining = pre.parse_known_args()
    config = load_yaml(known.config)
    defaults = config_section(config, "data")
    parser = build_parser(defaults)
    return parser.parse_args(["--config", known.config] + remaining if known.config else remaining)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    train_file = Path(args.train_file) if args.train_file else output_dir / "train.jsonl"
    eval_file = Path(args.eval_file) if args.eval_file else output_dir / "eval.jsonl"

    seen: set[str] = set()
    samples: list[dict[str, str]] = []

    samples.extend(take_local_self_samples(args.self_traces, args.self_count, seen, args))
    samples.extend(take_competition_samples(args.competition_jsonl, args.competition_count, seen, args))
    samples.extend(
        take_hf_samples(
            dataset_name=args.metamath_dataset,
            split=args.metamath_split,
            source="metamathqa",
            target_count=args.metamath_count,
            seen=seen,
            args=args,
        )
    )
    samples.extend(
        take_hf_samples(
            dataset_name=args.mathinstruct_dataset,
            split=args.mathinstruct_split,
            source="mathinstruct",
            target_count=args.mathinstruct_count,
            seen=seen,
            args=args,
        )
    )
    samples.extend(
        take_hf_samples(
            dataset_name=args.openmath_dataset,
            split=args.openmath_split,
            source="openmathinstruct2",
            target_count=args.openmath_count,
            seen=seen,
            args=args,
        )
    )

    if not samples:
        raise RuntimeError("No SFT samples were produced.")

    train_rows, eval_rows = split_train_eval(samples, args.eval_size, args.seed)
    write_jsonl(train_file, train_rows)
    write_jsonl(eval_file, eval_rows)

    source_counts = Counter(row["source"] for row in samples)
    print(json.dumps({"total": len(samples), "train": len(train_rows), "eval": len(eval_rows), "sources": source_counts}, indent=2))
    print(f"Wrote train data to {train_file}")
    print(f"Wrote eval data to {eval_file}")


if __name__ == "__main__":
    main()
