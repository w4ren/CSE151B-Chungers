# Added by Codex: model inference helpers; not part of the original starter repository.

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from math_comp.data import batched, is_mcq
from math_comp.prompts import build_prompt_text


def _torch_dtype(torch_module: Any, dtype_name: str | None) -> Any:
    if not dtype_name:
        return "auto"
    return getattr(torch_module, dtype_name)


def _generation_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "do_sample",
    }
    return {key: value for key, value in config.items() if key in allowed}


def generate_transformers(items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_config = config.get("model", {})
    gen_config = config.get("generation", {})
    prompt_config = config.get("prompt", {})

    model_id = model_config.get("id", "Qwen/Qwen3-4B-Thinking-2507")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
        "device_map": model_config.get("device_map", "auto"),
    }

    quantization = str(model_config.get("quantization", "4bit")).lower()
    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_torch_dtype(torch, model_config.get("torch_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=True,
        )
    elif quantization in {"none", "false", "no"}:
        model_kwargs["dtype"] = _torch_dtype(torch, model_config.get("torch_dtype"))
    else:
        raise ValueError(f"Unsupported transformers quantization: {quantization}")

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    adapter_dir = model_config.get("adapter_dir")
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    batch_size = int(gen_config.get("batch_size", 1))
    max_input_tokens = int(model_config.get("max_input_tokens", 16384))
    generation_kwargs = _generation_kwargs(gen_config)
    generation_kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

    records: list[dict[str, Any]] = []
    for batch in tqdm(list(batched(items, batch_size)), desc="Generating"):
        prompts = [build_prompt_text(tokenizer, item, prompt_config) for item in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        input_device = next(model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        prompt_width = inputs["input_ids"].shape[1]
        for item, output in zip(batch, output_ids):
            new_tokens = output[prompt_width:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            records.append({"id": int(item["id"]), "is_mcq": is_mcq(item), "response": response})

    return records


def generate_vllm(items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_config = config.get("model", {})
    gen_config = config.get("generation", {})
    prompt_config = config.get("prompt", {})
    vllm_config = config.get("vllm", {})
    model_id = model_config.get("id", "Qwen/Qwen3-4B-Thinking-2507")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    prompts = [build_prompt_text(tokenizer, item, prompt_config) for item in items]

    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
        "dtype": vllm_config.get("dtype", model_config.get("torch_dtype", "float16")),
        "load_format": vllm_config.get("load_format", "auto"),
        "gpu_memory_utilization": float(vllm_config.get("gpu_memory_utilization", 0.90)),
        "max_model_len": int(vllm_config.get("max_model_len", model_config.get("max_input_tokens", 4096))),
        "max_num_seqs": int(vllm_config.get("max_num_seqs", 16)),
        "max_num_batched_tokens": int(vllm_config.get("max_num_batched_tokens", 8192)),
        "enable_prefix_caching": bool(vllm_config.get("enable_prefix_caching", True)),
        "tensor_parallel_size": int(vllm_config.get("tensor_parallel_size", 1)),
    }
    quantization = vllm_config.get("quantization")
    if quantization and str(quantization).lower() not in {"none", "false", "no"}:
        llm_kwargs["quantization"] = quantization
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        max_tokens=int(gen_config.get("max_new_tokens", 32768)),
        temperature=float(gen_config.get("temperature", 0.6)),
        top_p=float(gen_config.get("top_p", 0.95)),
        top_k=int(gen_config.get("top_k", 20)),
        repetition_penalty=float(gen_config.get("repetition_penalty", 1.0)),
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    return [
        {"id": int(item["id"]), "is_mcq": is_mcq(item), "response": output.outputs[0].text.strip()}
        for item, output in zip(items, outputs)
    ]


def generate_responses(items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    backend = str(config.get("model", {}).get("backend", "transformers")).lower()
    if backend == "transformers":
        return generate_transformers(items, config)
    if backend == "vllm":
        return generate_vllm(items, config)
    raise ValueError(f"Unsupported backend: {backend}")
