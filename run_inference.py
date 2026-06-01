"""Single entry point for the competition inference pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from inference_pipeline.io import validate_predictions, write_submission_csv
from inference_pipeline.pipeline import PipelineConfig, run_model_pipeline


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = REPO_ROOT / "data/private.jsonl"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "submissions/final_submission.csv"
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


def run_inference(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    *,
    model_id: str | None = None,
    frq_adapter: str | Path = DEFAULT_FRQ_ADAPTER,
    rebuilt_adapter: str | Path = DEFAULT_REBUILT_ADAPTER,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 32768,
    vllm_batch_size: int = 16,
    predictions_path: str | Path | None = None,
) -> str:
    """Run end-to-end inference and write an ``id,response`` CSV.

    ``predictions_path`` is only for local validation of an already-generated
    JSONL. Normal use leaves it unset, which loads Qwen plus the two submitted
    LoRA adapters and runs the full generation/finalization pipeline.
    """

    data_path = _resolve(data_path)
    output_csv = _resolve(output_csv)

    if predictions_path is None:
        config = PipelineConfig(
            data_path=data_path,
            results_dir=_resolve(results_dir),
            model_id=model_id,
            frq_adapter=_adapter_ref(frq_adapter),
            rebuilt_adapter=_adapter_ref(rebuilt_adapter),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            vllm_batch_size=vllm_batch_size,
        )
        predictions_path = run_model_pipeline(config)
    else:
        predictions_path = _resolve(predictions_path)

    validate_predictions(data_path, predictions_path)
    write_submission_csv(data_path, predictions_path, output_csv)
    return str(output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and produce the final submission CSV.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH.relative_to(REPO_ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_CSV.relative_to(REPO_ROOT)))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--frq-adapter", default=DEFAULT_FRQ_ADAPTER)
    parser.add_argument("--rebuilt-adapter", default=DEFAULT_REBUILT_ADAPTER)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-batch-size", type=int, default=16)
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional generated predictions JSONL for local CSV validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_inference(
        data_path=args.data,
        output_csv=args.output,
        model_id=args.model_id,
        frq_adapter=args.frq_adapter,
        rebuilt_adapter=args.rebuilt_adapter,
        results_dir=args.results_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        vllm_batch_size=args.vllm_batch_size,
        predictions_path=args.predictions,
    )
    print(f"Wrote submission CSV to {output}")


if __name__ == "__main__":
    main()
