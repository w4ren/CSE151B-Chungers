# Added by Codex: prompt sweep experiment runner; not part of the original starter repository.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from math_comp.data import batched, is_mcq, load_yaml_config, read_jsonl
from math_comp.prompts import build_prompt_text
from math_comp.scoring import extract_answer_key
from math_comp.variants import get_variant, list_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt variants and sampled generations.")
    parser.add_argument("--input", default="data/public.jsonl", help="Input competition JSONL.")
    parser.add_argument("--output", default="codex_added/results/sweep_public.jsonl", help="Output JSONL.")
    parser.add_argument("--config", default="codex_added/configs/baseline.yaml", help="Base YAML config.")
    parser.add_argument("--variants", default="starter_deep,answer_audit", help="Comma-separated prompt variants.")
    parser.add_argument("--num-samples", type=int, default=4, help="Samples per item per prompt variant.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of input problems.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N input problems.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override generation batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max generated tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Override nucleus sampling p.")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k sampling.")
    parser.add_argument("--do-sample", action="store_true", help="Enable stochastic sampling.")
    parser.add_argument("--quantization", default=None, help="Override model quantization, e.g. none or 4bit.")
    parser.add_argument("--adapter-dir", default=None, help="Optional PEFT/LoRA adapter directory.")
    parser.add_argument(
        "--assistant-mode",
        choices=["think", "direct"],
        default=None,
        help="Use normal forced-thinking chat prompt or a direct assistant prompt.",
    )
    parser.add_argument(
        "--answer-key-mode",
        choices=["strict", "loose"],
        default="strict",
        help="Use explicit answers only, or allow the judger's loose fallback for voting keys.",
    )
    parser.add_argument("--assistant-prefix", default=None, help="Optional assistant text prefix to append to the prompt.")
    parser.add_argument("--resume", action="store_true", help="Append and skip completed id/variant/sample rows.")
    parser.add_argument("--list-variants", action="store_true", help="List prompt variants and exit.")
    return parser.parse_args()


def update_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model_config = config.setdefault("model", {})
    gen_config = config.setdefault("generation", {})
    model_config["backend"] = "transformers"

    if args.quantization is not None:
        model_config["quantization"] = args.quantization
    if args.adapter_dir is not None:
        model_config["adapter_dir"] = args.adapter_dir
    if args.batch_size is not None:
        gen_config["batch_size"] = args.batch_size
    if args.max_new_tokens is not None:
        gen_config["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        gen_config["temperature"] = args.temperature
    if args.top_p is not None:
        gen_config["top_p"] = args.top_p
    if args.top_k is not None:
        gen_config["top_k"] = args.top_k
    if args.do_sample:
        gen_config["do_sample"] = True
    if args.assistant_mode:
        config.setdefault("prompt", {})["assistant_mode"] = args.assistant_mode
    if args.assistant_prefix:
        config.setdefault("prompt", {})["assistant_prefix"] = args.assistant_prefix

    return config


def generation_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    gen_config = config.get("generation", {})
    keys = ["max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "do_sample"]
    return {key: gen_config[key] for key in keys if key in gen_config}


def load_transformers(config: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config.get("model", {})
    model_id = model_config.get("id", "Qwen/Qwen3-4B-Thinking-2507")
    trust_remote_code = bool(model_config.get("trust_remote_code", True))

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization = str(model_config.get("quantization", "none")).lower()
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "device_map": model_config.get("device_map", "auto"),
    }

    if quantization == "4bit":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("4-bit loading requires bitsandbytes support. Use --quantization none first.") from exc

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, model_config.get("torch_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=True,
        )
    elif quantization in {"none", "false", "no"}:
        dtype_name = model_config.get("torch_dtype", "bfloat16")
        model_kwargs["dtype"] = getattr(torch, dtype_name) if dtype_name else "auto"
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    adapter_dir = model_config.get("adapter_dir")
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model


def completed_keys(path: Path) -> set[tuple[int, str, int]]:
    if not path.exists():
        return set()
    done: set[tuple[int, str, int]] = set()
    for record in read_jsonl(path):
        done.add((int(record["id"]), str(record["variant"]), int(record["sample_index"])))
    return done


def main() -> None:
    args = parse_args()
    if args.list_variants:
        print("\n".join(list_variants()))
        return

    config = update_config(load_yaml_config(args.config), args)
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    variants = {name: get_variant(name) for name in variant_names}

    items = read_jsonl(args.input)
    if args.offset:
        items = items[args.offset :]
    if args.limit is not None:
        items = items[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"

    tasks: list[dict[str, Any]] = []
    for item in items:
        for variant_name in variant_names:
            for sample_index in range(args.num_samples):
                key = (int(item["id"]), variant_name, sample_index)
                if key not in done:
                    tasks.append({"item": item, "variant": variant_name, "sample_index": sample_index})

    print(f"Generating {len(tasks)} traces from {len(items)} problems and {len(variant_names)} variants.")
    tokenizer, model = load_transformers(config)
    gen_kwargs = generation_kwargs(config)
    gen_kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)
    batch_size = int(config.get("generation", {}).get("batch_size", 1))
    max_input_tokens = int(config.get("model", {}).get("max_input_tokens", 16384))

    import torch

    with out_path.open(mode, encoding="utf-8") as handle:
        for batch in tqdm(list(batched(tasks, batch_size)), desc="Prompt sweep"):
            prompts = [
                build_prompt_text(tokenizer, task["item"], variants[task["variant"]])
                for task in batch
            ]
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
                outputs = model.generate(**inputs, **gen_kwargs)

            prompt_width = inputs["input_ids"].shape[1]
            for task, output in zip(batch, outputs):
                item = task["item"]
                new_tokens = output[prompt_width:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                assistant_prefix = str(config.get("prompt", {}).get("assistant_prefix") or "")
                if assistant_prefix:
                    response = assistant_prefix + response
                generated_tokens = int(new_tokens.numel())
                record = {
                    "id": int(item["id"]),
                    "is_mcq": is_mcq(item),
                    "variant": task["variant"],
                    "sample_index": int(task["sample_index"]),
                    "answer_key": extract_answer_key(item, response, strict=args.answer_key_mode == "strict"),
                    "generated_tokens": generated_tokens,
                    "hit_token_limit": generated_tokens >= int(gen_kwargs.get("max_new_tokens", generated_tokens + 1)),
                    "response": response,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    print(f"Wrote sweep traces to {out_path}")


if __name__ == "__main__":
    main()
