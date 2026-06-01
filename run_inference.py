"""Single entry point for the competition inference pipeline."""

from __future__ import annotations

from pathlib import Path

from inference_pipeline.io import validate_predictions, write_submission_csv
from inference_pipeline.pipeline import PipelineConfig, run_model_pipeline


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = REPO_ROOT / "data/private.jsonl"
DEFAULT_OUTPUT_CSV_NAME = "final_submission.csv"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results/inference"
DEFAULT_FRQ_ADAPTER = "wren88/cse151b-chungers-frq-finalizer-lora"
DEFAULT_REBUILT_ADAPTER = "wren88/cse151b-chungers-rebuilt-answer-lora"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _adapter_ref(value: str | Path) -> str | Path:
    """Return local adapter paths as paths and Hub repo IDs as strings."""

    if isinstance(value, Path):
        return _resolve(value)
    text = str(value)
    local_candidate = _resolve(text)
    if local_candidate.exists() or text.startswith((".", "/")):
        return local_candidate
    return text


def run_inference() -> str:
    """Run end-to-end inference and write ``final_submission.csv`` to cwd."""

    data_path = DEFAULT_DATA_PATH
    output_csv = Path.cwd() / DEFAULT_OUTPUT_CSV_NAME
    config = PipelineConfig(
        data_path=data_path,
        results_dir=DEFAULT_RESULTS_DIR,
        model_id=None,
        frq_adapter=_adapter_ref(DEFAULT_FRQ_ADAPTER),
        rebuilt_adapter=_adapter_ref(DEFAULT_REBUILT_ADAPTER),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=32768,
        vllm_batch_size=16,
    )
    predictions_path = run_model_pipeline(config)

    validate_predictions(data_path, predictions_path)
    write_submission_csv(data_path, predictions_path, output_csv)
    return str(output_csv)


def main() -> None:
    output = run_inference()
    print(f"Wrote submission CSV to {output}")


if __name__ == "__main__":
    main()
