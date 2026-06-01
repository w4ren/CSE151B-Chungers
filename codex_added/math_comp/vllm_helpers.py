# Added by Codex: shared vLLM helpers; not part of the original starter repository.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def add_vllm_cli_args(parser: argparse.ArgumentParser, *, default_backend: str = "transformers") -> None:
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default=default_backend,
        help="Generation backend. vLLM is fastest when .venv-vllm is active.",
    )
    parser.add_argument("--dtype", default="float16", help="vLLM model dtype; float16 is fastest on RTX 2080 Ti.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM GPU memory fraction.")
    parser.add_argument("--max-model-len", type=int, default=None, help="vLLM max context length.")
    parser.add_argument("--max-num-seqs", type=int, default=16, help="vLLM scheduler sequence concurrency.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192, help="vLLM scheduler token budget.")
    parser.add_argument("--vllm-batch-size", type=int, default=1, help="Prompt chunk size for vLLM; 0 means all prompts.")
    parser.add_argument("--vllm-quantization", default=None, help="Optional vLLM quantization, e.g. bitsandbytes.")
    parser.add_argument("--vllm-load-format", default="auto", help="Optional vLLM load format.")
    parser.add_argument("--vllm-max-lora-rank", type=int, default=64, help="Maximum LoRA rank for vLLM adapters.")
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="Optional vLLM attention backend, e.g. FLASHINFER, FLASH_ATTN, or TRITON_ATTN.",
    )
    parser.add_argument("--enforce-eager", action="store_true", help="Skip CUDA graph compilation for faster startup.")
    parser.add_argument("--safetensors-load-strategy", default=None, help="Optional vLLM checkpoint loading strategy, e.g. prefetch.")
    parser.add_argument("--disable-prefix-caching", action="store_true", help="Disable vLLM prefix caching.")
    parser.add_argument("--disable-async-scheduling", action="store_true", help="Disable vLLM async scheduling.")


def make_sampling_params(
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    do_sample: bool = True,
    allowed_token_ids: list[int] | None = None,
) -> Any:
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "temperature": float(temperature) if do_sample else 0.0,
        "top_p": float(top_p),
        "repetition_penalty": float(repetition_penalty),
    }
    if top_k is not None:
        kwargs["top_k"] = int(top_k)
    if allowed_token_ids is not None:
        kwargs["allowed_token_ids"] = list(allowed_token_ids)
    return SamplingParams(**kwargs)


def make_llm(model_id: str, args: argparse.Namespace, *, trust_remote_code: bool = True, adapter_dir: str | None = None) -> Any:
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": model_id,
        "trust_remote_code": trust_remote_code,
        "dtype": args.dtype,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_num_seqs": int(args.max_num_seqs),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "enable_prefix_caching": not bool(args.disable_prefix_caching),
    }
    if getattr(args, "disable_async_scheduling", False):
        kwargs["async_scheduling"] = False
    if args.max_model_len is not None:
        kwargs["max_model_len"] = int(args.max_model_len)
    if adapter_dir and (Path(adapter_dir) / "tokenizer_config.json").exists():
        kwargs["tokenizer"] = str(Path(adapter_dir))
    attention_backend = getattr(args, "attention_backend", None)
    if attention_backend:
        kwargs["attention_config"] = {"backend": str(attention_backend).upper()}
    if getattr(args, "enforce_eager", False):
        kwargs["enforce_eager"] = True
    load_strategy = getattr(args, "safetensors_load_strategy", None)
    if load_strategy:
        kwargs["safetensors_load_strategy"] = load_strategy
    if args.vllm_quantization and str(args.vllm_quantization).lower() not in {"none", "false", "no"}:
        kwargs["quantization"] = args.vllm_quantization
        kwargs["load_format"] = args.vllm_load_format
    elif args.vllm_load_format != "auto":
        kwargs["load_format"] = args.vllm_load_format
    if adapter_dir:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = int(args.vllm_max_lora_rank)

    return LLM(**kwargs)


def make_lora_request(adapter_dir: str | None, *, name: str = "adapter", int_id: int = 1) -> Any | None:
    if not adapter_dir:
        return None

    from vllm.lora.request import LoRARequest

    return LoRARequest(name, int_id, str(Path(adapter_dir)))


def vllm_generated_tokens(output: Any) -> int:
    token_ids = getattr(output.outputs[0], "token_ids", None)
    return len(token_ids or [])


def vllm_text(output: Any) -> str:
    return str(output.outputs[0].text).strip()
