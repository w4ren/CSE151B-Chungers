# Added by Codex: evaluate a direct MCQ LoRA with the same prompt used for public-MCQ SFT.

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

from judger import Judger
from math_comp.data import batched, has_gold, is_mcq, read_jsonl, write_jsonl
from math_comp.final_answer import normalize_final_response
from math_comp.grpo_finalizer import direct_chat_prompt
from math_comp.prompts import format_options
from math_comp.scoring import extract_answer_key, score_item, summarize_results
from math_comp.vllm_helpers import add_vllm_cli_args, make_llm, make_lora_request, make_sampling_params, vllm_generated_tokens, vllm_text


SYSTEM = (
    "You are an expert mathematician answering a multiple-choice math problem. "
    "Read the problem and options, then output exactly one option letter inside \\boxed{}. "
    "Do not output option text. Do not explain."
)


def default_model_id() -> str:
    snapshot_root = REPO_ROOT / ".hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots"
    snapshots = [path for path in snapshot_root.glob("*") if path.is_dir()]
    if snapshots:
        return str(max(snapshots, key=lambda path: path.stat().st_mtime))
    return "Qwen/Qwen3-4B-Thinking-2507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate direct MCQ LoRA on MCQ rows.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--model-id", default=default_model_id())
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--score", action="store_true")
    add_vllm_cli_args(parser, default_backend="vllm")
    return parser.parse_args()


def messages(item: dict[str, Any]) -> list[dict[str, str]]:
    user = (
        "Problem:\n"
        f"{item['question']}\n\n"
        "Options:\n"
        f"{format_options(item.get('options') or [])}\n\n"
        "Final answer only:"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def normalize_mcq(item: dict[str, Any], text: str) -> str:
    response = normalize_final_response(item, str(text).strip())
    key = extract_answer_key(item, response, strict=True)
    if key:
        return "\\boxed{" + key.strip().upper() + "}"
    key = extract_answer_key(item, text, strict=True)
    if key:
        return "\\boxed{" + key.strip().upper() + "}"
    return response


def main() -> None:
    args = parse_args()
    if args.backend != "vllm":
        raise ValueError("67_eval_direct_mcq_lora.py currently supports --backend vllm only.")

    data = [item for item in read_jsonl(args.data) if is_mcq(item)]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir if Path(args.adapter_dir).exists() else args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = make_llm(args.model_id, args, trust_remote_code=True, adapter_dir=args.adapter_dir)
    lora_request = make_lora_request(args.adapter_dir)
    sampling = make_sampling_params(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )
    prompts = [direct_chat_prompt(tokenizer, messages(item)) for item in data]
    chunk_size = int(args.vllm_batch_size or len(prompts) or 1)
    out_rows: list[dict[str, Any]] = []
    judger = Judger(strict_extract=False)

    for item_batch, prompt_batch in tqdm(list(zip(batched(data, chunk_size), batched(prompts, chunk_size))), desc="Direct MCQ eval"):
        outputs = llm.generate(list(prompt_batch), sampling_params=sampling, lora_request=lora_request)
        for item, output in zip(item_batch, outputs):
            raw_text = vllm_text(output)
            response = normalize_mcq(item, raw_text)
            row = {
                "id": int(item["id"]),
                "is_mcq": True,
                "raw_response": raw_text,
                "response": response,
                "answer_key": extract_answer_key(item, response, strict=True),
                "generated_tokens": vllm_generated_tokens(output),
            }
            if has_gold(item):
                row["gold"] = item["answer"]
                row["correct"] = score_item(judger, item, response)
            out_rows.append(row)

    write_jsonl(args.output, out_rows)
    if args.score and all("correct" in row for row in out_rows):
        print(json.dumps(summarize_results(out_rows), indent=2))
    print(f"Wrote {len(out_rows)} direct MCQ predictions to {args.output}")


if __name__ == "__main__":
    main()
