from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from inference_pipeline.formatting import answer_slot_count, extract_answer_key, is_mcq, normalize_final_response
from inference_pipeline.io import batched, index_by_id, read_jsonl, write_jsonl
from inference_pipeline.prompts import build_finalizer_prompt, build_raw_prompt


BASE_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


@dataclass
class PipelineConfig:
    data_path: Path
    results_dir: Path
    model_id: str | None
    frq_adapter: str | Path
    rebuilt_adapter: str | Path
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 32768
    vllm_batch_size: int = 16
    raw_token_ladder: tuple[int, ...] = (8192, 16384, 32768)
    raw_temperature: float = 0.6
    raw_top_p: float = 0.95
    raw_top_k: int = 20
    finalizer_max_tokens: int = 64
    finalizer_retry_tokens: int = 128
    finalizer_temperature: float = 0.1
    finalizer_top_p: float = 0.9
    max_trace_chars: int = 6000
    dtype: str = "float16"
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 8192
    extra_llm_kwargs: dict[str, Any] = field(default_factory=dict)


def default_model_id(repo_root: Path) -> str:
    snapshot_root = repo_root / ".hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots"
    snapshots = [path for path in snapshot_root.glob("*") if path.is_dir()]
    if snapshots:
        return str(max(snapshots, key=lambda path: path.stat().st_mtime))
    return BASE_MODEL_ID


def _materialize_adapter(adapter: str | Path) -> str:
    path = Path(adapter)
    if path.exists():
        return str(path)
    text = str(adapter)
    if path.is_absolute() or text.startswith("."):
        raise FileNotFoundError(f"Missing adapter directory: {adapter}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub or use a local adapter directory.") from exc
    return snapshot_download(repo_id=text)


def _load_vllm(config: PipelineConfig) -> tuple[Any, Any, Any, Any]:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise RuntimeError("Install the packages in requirements.txt before running inference.") from exc

    repo_root = Path(__file__).resolve().parents[1]
    model_id = config.model_id or default_model_id(repo_root)
    frq_adapter = _materialize_adapter(config.frq_adapter)
    rebuilt_adapter = _materialize_adapter(config.rebuilt_adapter)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "trust_remote_code": True,
        "dtype": config.dtype,
        "tensor_parallel_size": int(config.tensor_parallel_size),
        "gpu_memory_utilization": float(config.gpu_memory_utilization),
        "max_model_len": int(config.max_model_len),
        "max_num_seqs": int(config.max_num_seqs),
        "max_num_batched_tokens": int(config.max_num_batched_tokens),
        "enable_lora": True,
        "max_lora_rank": 64,
        "max_loras": 2,
    }
    llm_kwargs.update(config.extra_llm_kwargs)
    llm = LLM(**llm_kwargs)
    frq_request = LoRARequest("frq_finalizer", 1, frq_adapter)
    rebuilt_request = LoRARequest("rebuilt_answer", 2, rebuilt_adapter)
    return tokenizer, llm, frq_request, rebuilt_request


def _sampling_params(*, max_tokens: int, temperature: float, top_p: float, top_k: int | None = None) -> Any:
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if top_k is not None:
        kwargs["top_k"] = int(top_k)
    return SamplingParams(**kwargs)


def _generated_tokens(output: Any) -> int:
    token_ids = getattr(output.outputs[0], "token_ids", None)
    return len(token_ids or [])


def _text(output: Any) -> str:
    return str(output.outputs[0].text).strip()


def _generate(
    llm: Any,
    prompts: list[str],
    sampling_params: Any,
    *,
    batch_size: int,
    lora_request: Any | None = None,
    desc: str,
) -> list[Any]:
    outputs: list[Any] = []
    for chunk in tqdm(list(batched(prompts, batch_size)), desc=desc):
        outputs.extend(llm.generate(chunk, sampling_params=sampling_params, lora_request=lora_request))
    return outputs


def _run_raw_ladder(tokenizer: Any, llm: Any, items: list[dict[str, Any]], config: PipelineConfig) -> list[dict[str, Any]]:
    raw_by_id: dict[int, dict[str, Any]] = {}
    remaining = list(items)
    for max_tokens in config.raw_token_ladder:
        if not remaining:
            break
        prompts = [build_raw_prompt(tokenizer, item) for item in remaining]
        sampling = _sampling_params(
            max_tokens=max_tokens,
            temperature=config.raw_temperature,
            top_p=config.raw_top_p,
            top_k=config.raw_top_k,
        )
        outputs = _generate(
            llm,
            prompts,
            sampling,
            batch_size=config.vllm_batch_size,
            desc=f"Raw generation {max_tokens}",
        )
        next_remaining: list[dict[str, Any]] = []
        for item, output in zip(remaining, outputs):
            generated_tokens = _generated_tokens(output)
            response = _text(output)
            row = {
                "id": int(item["id"]),
                "is_mcq": is_mcq(item),
                "response": response,
                "answer_key": extract_answer_key(item, response),
                "generated_tokens": generated_tokens,
                "hit_token_limit": generated_tokens >= max_tokens,
                "raw_max_tokens": max_tokens,
            }
            raw_by_id[int(item["id"])] = row
            if row["hit_token_limit"] and max_tokens != config.raw_token_ladder[-1]:
                next_remaining.append(item)
        remaining = next_remaining
    return [raw_by_id[int(item["id"])] for item in items]


def _bad_format_reason(item: dict[str, Any], response: str) -> str | None:
    if is_mcq(item):
        return None
    expected = answer_slot_count(item)
    answer = extract_answer_key(item, response)
    parsed = 0 if not answer else len([part for part in answer.split(",") if part.strip()])
    if expected and parsed != expected:
        return f"it produced {parsed} answer field(s), but {expected} field(s) are required"
    return None


def _finalize_rows(
    tokenizer: Any,
    llm: Any,
    items: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    config: PipelineConfig,
    *,
    lora_request: Any,
    desc: str,
) -> list[dict[str, Any]]:
    data_by_id = index_by_id(items)
    prompts = [
        build_finalizer_prompt(
            tokenizer,
            data_by_id[int(row["id"])],
            str(row.get("response", "")),
            max_trace_chars=config.max_trace_chars,
        )
        for row in raw_rows
    ]
    sampling = _sampling_params(
        max_tokens=config.finalizer_max_tokens,
        temperature=config.finalizer_temperature,
        top_p=config.finalizer_top_p,
    )
    outputs = _generate(
        llm,
        prompts,
        sampling,
        batch_size=config.vllm_batch_size,
        lora_request=lora_request,
        desc=desc,
    )

    finalized: list[dict[str, Any]] = []
    retry_jobs: list[tuple[int, dict[str, Any], dict[str, Any], str]] = []
    for row, output in zip(raw_rows, outputs):
        item = data_by_id[int(row["id"])]
        response = normalize_final_response(item, _text(output))
        record = {
            "id": int(row["id"]),
            "is_mcq": is_mcq(item),
            "response": response,
            "answer_key": extract_answer_key(item, response),
            "source_raw_max_tokens": row.get("raw_max_tokens"),
            "source_hit_token_limit": row.get("hit_token_limit"),
        }
        reason = _bad_format_reason(item, response)
        if reason:
            retry_jobs.append((len(finalized), row, record, reason))
        finalized.append(record)

    if retry_jobs:
        retry_prompts = [
            build_finalizer_prompt(
                tokenizer,
                data_by_id[int(row["id"])],
                str(row.get("response", "")),
                max_trace_chars=config.max_trace_chars,
                retry_reason=reason,
                first_attempt=record["response"],
            )
            for _, row, record, reason in retry_jobs
        ]
        retry_sampling = _sampling_params(
            max_tokens=config.finalizer_retry_tokens,
            temperature=config.finalizer_temperature,
            top_p=config.finalizer_top_p,
        )
        retry_outputs = _generate(
            llm,
            retry_prompts,
            retry_sampling,
            batch_size=config.vllm_batch_size,
            lora_request=lora_request,
            desc=f"{desc} retry",
        )
        for (record_index, row, record, reason), output in zip(retry_jobs, retry_outputs):
            item = data_by_id[int(row["id"])]
            retry_response = normalize_final_response(item, _text(output))
            finalized[record_index] = {
                **record,
                "response": retry_response,
                "answer_key": extract_answer_key(item, retry_response),
                "first_finalizer_response": record["response"],
                "retry_reason": reason,
            }

    return finalized


def run_model_pipeline(config: PipelineConfig) -> Path:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    items = read_jsonl(config.data_path)
    tokenizer, llm, frq_request, rebuilt_request = _load_vllm(config)

    raw_rows = _run_raw_ladder(tokenizer, llm, items, config)
    raw_path = config.results_dir / "raw_generation.jsonl"
    write_jsonl(raw_path, raw_rows)

    data_by_id = index_by_id(items)
    frq_raw = [row for row in raw_rows if not is_mcq(data_by_id[int(row["id"])])]
    mcq_raw = [row for row in raw_rows if is_mcq(data_by_id[int(row["id"])])]

    frq_items = [data_by_id[int(row["id"])] for row in frq_raw]
    mcq_items = [data_by_id[int(row["id"])] for row in mcq_raw]
    frq_final = _finalize_rows(
        tokenizer,
        llm,
        frq_items,
        frq_raw,
        config,
        lora_request=frq_request,
        desc="FRQ finalizer",
    )
    mcq_final = _finalize_rows(
        tokenizer,
        llm,
        mcq_items,
        mcq_raw,
        config,
        lora_request=rebuilt_request,
        desc="MCQ finalizer",
    )

    final_by_id = index_by_id(frq_final + mcq_final)
    final_rows = [{"id": int(item["id"]), "response": final_by_id[int(item["id"])]["response"]} for item in items]
    predictions_path = config.results_dir / "predictions.jsonl"
    write_jsonl(predictions_path, final_rows)
    return predictions_path
