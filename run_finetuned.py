"""
CSE 151B Spring 2026 — Inference with a fine-tuned LoRA adapter
Loads the base Qwen3-4B-Thinking + a LoRA adapter from train.py, then runs inference.

Usage:
    python run_finetuned.py --adapter ./checkpoints/numina-qlora/final
                            [--data data/public.jsonl]
                            [--output results/finetuned_results.jsonl]
                            [--no-eval]

If --adapter is not given, falls back to plain base-model inference (same as baseline.py).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH = "results/finetuned_results.jsonl"
GPU_ID      = "0"
MAX_TOKENS  = 8192

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def extract_answer(text: str) -> str:
    clean   = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    matches = re.findall(r"\\boxed\{([^}]+)\}", clean)
    if matches:
        return matches[-1].strip()
    lines = [l.strip() for l in clean.split("\n") if l.strip()]
    return lines[-1] if lines else clean


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


def acc(subset: list) -> float:
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0


def evaluate(results: list) -> None:
    scored   = [r for r in results if "correct" in r]
    mcq_res  = [r for r in scored if r["is_mcq"]]
    free_res = [r for r in scored if not r["is_mcq"]]
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {sum(r['correct'] for r in scored):4d} / {len(scored):4d}  ({acc(scored):.2f}%)")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Inference with optional LoRA adapter")
    parser.add_argument("--adapter",    default=None,        help="Path to LoRA adapter dir (from train.py)")
    parser.add_argument("--base-model", default=MODEL_ID,    help="Base model ID or path")
    parser.add_argument("--data",       default=DATA_PATH,   help="Path to input .jsonl")
    parser.add_argument("--output",     default=OUTPUT_PATH, help="Path to output .jsonl")
    parser.add_argument("--gpu",        default=GPU_ID,      help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--max-tokens", default=MAX_TOKENS,  type=int)
    parser.add_argument("--limit",      default=None,        type=int)
    parser.add_argument("--no-eval",    action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading data from {args.data} ...")
    data = [json.loads(line) for line in open(args.data)]
    if args.limit:
        data = data[: args.limit]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading model {args.base_model} ...")
    llm_kwargs = dict(
        model=args.base_model,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=8172,
    )
    if args.adapter:
        # vLLM supports LoRA adapters natively
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64
        print(f"LoRA adapter : {args.adapter}")

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )

    tokenizer = llm.get_tokenizer()
    print("Model loaded.")

    # ── Build prompts ──────────────────────────────────────────────────────────
    MAX_PROMPT_TOKENS = 16384 - args.max_tokens - 64
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer.encode(prompt_text)
        if len(token_ids) > MAX_PROMPT_TOKENS:
            token_ids = token_ids[:MAX_PROMPT_TOKENS]
            prompt_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        prompts.append(prompt_text)

    # ── Generate ───────────────────────────────────────────────────────────────
    print(f"Generating responses for {len(prompts)} questions...")
    gen_kwargs = {}
    if args.adapter:
        gen_kwargs["lora_request"] = LoRARequest("math_adapter", 1, args.adapter)

    outputs   = llm.generate(prompts, sampling_params, **gen_kwargs)
    responses = [out.outputs[0].text.strip() for out in outputs]

    # ── Score ──────────────────────────────────────────────────────────────────
    results = []
    if not args.no_eval:
        sys.path.insert(0, ".")
        from judger import Judger
        judger = Judger(strict_extract=False)

        for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
            is_mcq = bool(item.get("options"))
            gold   = item["answer"]
            if is_mcq:
                correct = score_mcq(response, str(gold))
            else:
                gold_list = gold if isinstance(gold, list) else [gold]
                try:
                    correct = judger.auto_judge(
                        pred=response, gold=gold_list, options=[[]] * len(gold_list)
                    )
                except Exception:
                    correct = False
            results.append({
                "id": item.get("id"), "is_mcq": is_mcq,
                "gold": gold, "response": response, "correct": correct,
            })
    else:
        for item, response in zip(data, responses):
            results.append({
                "id": item.get("id"), "is_mcq": bool(item.get("options")), "response": response,
            })

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} records to {out_path}")

    if not args.no_eval:
        evaluate(results)


if __name__ == "__main__":
    main()
