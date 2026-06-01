# Added by Codex: dataset and config helpers; not part of the original starter repository.

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


JsonRecord = dict[str, Any]


def read_jsonl(path: str | Path) -> list[JsonRecord]:
    records: list[JsonRecord] = []
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


def read_jsonl_complete_prefix(path: str | Path) -> list[JsonRecord]:
    records: list[JsonRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return records


def _fsync_parent_dir(path: Path) -> None:
    try:
        fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_jsonl(path: str | Path, records: Iterable[JsonRecord]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
        _fsync_parent_dir(out_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def append_jsonl_record(path: str | Path, record: JsonRecord) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_parent_dir(out_path)


def load_yaml_config(path: str | Path) -> JsonRecord:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML or run `pip install -r requirements.txt`.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def is_mcq(item: JsonRecord) -> bool:
    return bool(item.get("options"))


def has_gold(item: JsonRecord) -> bool:
    return "answer" in item and item["answer"] not in (None, "")


def answer_slot_count(item: JsonRecord) -> int:
    if is_mcq(item):
        return 1
    answer = item.get("answer")
    if isinstance(answer, list):
        return len(answer)
    marker_count = str(item.get("question", "")).count("[ANS]")
    return marker_count or 1


def summarize_dataset(items: Sequence[JsonRecord]) -> JsonRecord:
    slot_counts = Counter(answer_slot_count(item) for item in items)
    mcq = sum(is_mcq(item) for item in items)
    with_gold = sum(has_gold(item) for item in items)
    return {
        "total": len(items),
        "mcq": mcq,
        "free_form": len(items) - mcq,
        "with_gold": with_gold,
        "without_gold": len(items) - with_gold,
        "answer_slots": dict(sorted(slot_counts.items())),
    }


def batched(items: Sequence[JsonRecord], batch_size: int) -> Iterator[list[JsonRecord]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def index_by_id(items: Iterable[JsonRecord]) -> dict[int, JsonRecord]:
    indexed: dict[int, JsonRecord] = {}
    for item in items:
        item_id = int(item["id"])
        if item_id in indexed:
            raise ValueError(f"Duplicate id found: {item_id}")
        indexed[item_id] = item
    return indexed
