from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, TypeVar


JsonRecord = dict[str, Any]
T = TypeVar("T")


def read_jsonl(path: str | Path) -> list[JsonRecord]:
    records: list[JsonRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


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
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def index_by_id(records: Iterable[JsonRecord]) -> dict[int, JsonRecord]:
    by_id: dict[int, JsonRecord] = {}
    for record in records:
        item_id = int(record["id"])
        if item_id in by_id:
            raise ValueError(f"Duplicate id found: {item_id}")
        by_id[item_id] = record
    return by_id


def batched(items: list[T], batch_size: int) -> Iterable[list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def validate_predictions(data_path: str | Path, predictions_path: str | Path) -> None:
    data = read_jsonl(data_path)
    predictions = read_jsonl(predictions_path)
    pred_by_id = index_by_id(predictions)
    data_ids = [int(item["id"]) for item in data]
    data_id_set = set(data_ids)
    pred_id_set = set(pred_by_id)

    missing = [item_id for item_id in data_ids if item_id not in pred_id_set]
    extra = sorted(pred_id_set - data_id_set)
    empty = [item_id for item_id in data_ids if not str(pred_by_id[item_id].get("response", "")).strip()]
    if missing or extra or empty:
        raise ValueError(
            "Invalid prediction JSONL: "
            f"missing={missing[:20]}, extra={extra[:20]}, empty_response={empty[:20]}"
        )


def write_submission_csv(data_path: str | Path, predictions_path: str | Path, output_path: str | Path) -> None:
    data = read_jsonl(data_path)
    predictions = index_by_id(read_jsonl(predictions_path))
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "response"])
        writer.writeheader()
        for item in data:
            item_id = int(item["id"])
            writer.writerow({"id": item_id, "response": str(predictions[item_id]["response"])})
