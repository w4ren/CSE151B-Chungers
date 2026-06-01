"""Single entry point for producing the final competition submission CSV.

The heavy model generations and adapter finalization stages write a final
prediction JSONL with one ``{"id": ..., "response": ...}`` record per private
row. This entry point performs the final reproducible routing validation and
CSV emission used for the submitted file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
CODEX_ROOT = REPO_ROOT / "codex_added"
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from math_comp.data import index_by_id, read_jsonl  # noqa: E402
from math_comp.submission import write_submission_csv  # noqa: E402


DEFAULT_DATA_PATH = REPO_ROOT / "data/private.jsonl"
DEFAULT_PREDICTIONS_PATH = (
    REPO_ROOT / "codex_added/results/final_submission_source_20260601/combined_predictions_full_private_943.jsonl"
)
DEFAULT_OUTPUT_CSV = REPO_ROOT / "codex_added/submissions/final_submission.csv"
DEFAULT_GENERATION_OUT_DIR = REPO_ROOT / "codex_added/results/run_inference_generated"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def validate_prediction_source(data_path: str | Path, predictions_path: str | Path) -> dict[str, Any]:
    """Validate that predictions exactly cover the private ids and have responses."""

    data_path = _resolve(data_path)
    predictions_path = _resolve(predictions_path)
    data = read_jsonl(data_path)
    predictions = read_jsonl(predictions_path)
    pred_by_id = index_by_id(predictions)

    data_ids = [int(row["id"]) for row in data]
    data_id_set = set(data_ids)
    pred_id_set = set(pred_by_id)
    missing = [item_id for item_id in data_ids if item_id not in pred_id_set]
    extra = sorted(pred_id_set - data_id_set)
    empty_response = [item_id for item_id in data_ids if not str(pred_by_id[item_id].get("response", ""))]

    if missing or extra or empty_response:
        details = {
            "missing": missing[:20],
            "extra": extra[:20],
            "empty_response": empty_response[:20],
        }
        raise ValueError(f"Invalid prediction source for submission: {json.dumps(details, sort_keys=True)}")

    return {
        "data_path": str(data_path.relative_to(REPO_ROOT)),
        "predictions_path": str(predictions_path.relative_to(REPO_ROOT)),
        "rows": len(data_ids),
        "first_id": data_ids[0] if data_ids else None,
        "last_id": data_ids[-1] if data_ids else None,
    }


def run_inference(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    predictions_path: str | Path = DEFAULT_PREDICTIONS_PATH,
    generate_if_missing: bool = True,
    generation_out_dir: str | Path = DEFAULT_GENERATION_OUT_DIR,
) -> str:
    """Write the final competition CSV and return its path.

    Parameters are path-like so Gradescope or a local runner can supply a
    private JSONL path and desired output path without editing this file.
    """

    data_path = _resolve(data_path)
    output_csv = _resolve(output_csv)
    predictions_path = _resolve(predictions_path)

    if not predictions_path.exists():
        if not generate_if_missing:
            raise FileNotFoundError(f"Missing final prediction source: {predictions_path}")
        generation_out_dir = _resolve(generation_out_dir)
        script = REPO_ROOT / "codex_added/scripts/19_run_best_pipeline.sh"
        if not script.exists():
            raise FileNotFoundError(f"Missing generation script: {script}")
        command = ["bash", str(script), str(data_path), str(generation_out_dir), str(output_csv)]
        print("Final prediction source not found; running model pipeline:")
        print(" ".join(command))
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        return str(output_csv)

    summary = validate_prediction_source(data_path, predictions_path)
    write_submission_csv(data_path, predictions_path, output_csv)
    summary["output_csv"] = str(output_csv.relative_to(REPO_ROOT))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return str(output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce the final id,response submission CSV.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH.relative_to(REPO_ROOT)))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS_PATH.relative_to(REPO_ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_CSV.relative_to(REPO_ROOT)))
    parser.add_argument("--generation-out-dir", default=str(DEFAULT_GENERATION_OUT_DIR.relative_to(REPO_ROOT)))
    parser.add_argument(
        "--no-generate-if-missing",
        action="store_true",
        help="Fail instead of invoking the model pipeline when the final prediction JSONL is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        data_path=args.data,
        output_csv=args.output,
        predictions_path=args.predictions,
        generate_if_missing=not args.no_generate_if_missing,
        generation_out_dir=args.generation_out_dir,
    )


if __name__ == "__main__":
    main()
