# Added by Codex: submission CSV writer; not part of the original starter repository.

from __future__ import annotations

import csv
from pathlib import Path

from math_comp.data import index_by_id, read_jsonl


def write_submission_csv(data_path: str | Path, predictions_path: str | Path, output_path: str | Path) -> None:
    data = read_jsonl(data_path)
    predictions = read_jsonl(predictions_path)
    pred_by_id = index_by_id(predictions)

    missing = [int(item["id"]) for item in data if int(item["id"]) not in pred_by_id]
    if missing:
        preview = ", ".join(str(item_id) for item_id in missing[:10])
        raise ValueError(f"Missing predictions for {len(missing)} ids: {preview}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "response"])
        writer.writeheader()
        for item in data:
            item_id = int(item["id"])
            writer.writerow({"id": item_id, "response": str(pred_by_id[item_id].get("response", ""))})
