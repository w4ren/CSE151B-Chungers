"""
CSE 151B Spring 2026 — Math Reasoning Competition
Token-budget sweep: runs 20 questions at each of 5 max-token limits.

Usage:
    python multi_token_budget.py [--data DATA_PATH] [--output-dir OUTPUT_DIR]
                                 [--gpu GPU_ID] [--no-eval]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH     = "data/public.jsonl"
OUTPUT_DIR    = "results/token_budget_sweep"
GPU_ID        = "0"
N_QUESTIONS   = 20
TOKEN_BUDGETS = [1024, 2048, 4096, 8192, 16384, 32768]
MAX_MODEL_LEN = 16000   # 32768 gen + ~4 K prompt headroom

# ── Prompts ───────────────────────────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def plot_summary(summary: list, out_dir: Path, n_questions: int) -> None:
    budgets  = [s[0] for s in summary]
    overall  = [s[1] for s in summary]
    mcq      = [s[2] for s in summary]
    free     = [s[3] for s in summary]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(budgets, overall, marker="o", linewidth=2, label="Overall")
    ax.plot(budgets, mcq,     marker="s", linewidth=2, label="MCQ")
    ax.plot(budgets, free,    marker="^", linewidth=2, label="Free-form")

    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Max tokens (generation budget)", fontsize=12)
    ax.set_ylabel(f"Accuracy % (out of {n_questions})", fontsize=12)
    ax.set_title("Accuracy vs. Token Budget", fontsize=14)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plot_path = out_dir / "accuracy_vs_tokens.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {plot_path}")


def score_results(data: list, responses: list, judger) -> list:
    results = []
    for item, response in zip(data, responses):
        is_mcq = bool(item.get("options"))
        gold   = item["answer"]

        if is_mcq:
            correct = score_mcq(response, str(gold))
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        results.append({
            "id":       item.get("id"),
            "is_mcq":   is_mcq,
            "gold":     gold,
            "response": response,
            "correct":  correct,
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Token-budget sweep inference script")
    parser.add_argument("--data",       default=DATA_PATH,  help="Path to input .jsonl file")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for per-budget result files")
    parser.add_argument("--gpu",        default=GPU_ID,     help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--no-eval",    action="store_true", help="Skip scoring")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ── 1. Load dataset ────────────────────────────────────────────────────────
    print(f"Loading data from {args.data} ...")
    data = [json.loads(line) for line in open(args.data)][:N_QUESTIONS]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # ── 2. Load model (once) ───────────────────────────────────────────────────
    print(f"Loading model {MODEL_ID} ...")
    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=8172,
    )
    tokenizer = llm.get_tokenizer()
    print("Model loaded.\n")

    # ── 3. Build prompts (shared across all budgets) ───────────────────────────
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
    print(f"Built {len(prompts)} prompts. Sample length: {len(tokenizer.encode(prompts[0]))} tokens.")

    # ── 4. Optionally load judger ──────────────────────────────────────────────
    judger = None
    if not args.no_eval:
        sys.path.insert(0, ".")
        from judger import Judger
        judger = Judger(strict_extract=False)

    # ── 5. Sweep over token budgets ────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []   # (budget, overall_acc, mcq_acc, free_acc)

    for budget in TOKEN_BUDGETS:
        print(f"{'='*60}")
        print(f"  max_tokens = {budget}")
        print(f"{'='*60}")

        sampling_params = SamplingParams(
            max_tokens=budget,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
        )

        outputs   = llm.generate(prompts, sampling_params)
        responses = [out.outputs[0].text.strip() for out in outputs]

        if not args.no_eval:
            results = score_results(data, responses, judger)

            mcq_res  = [r for r in results if r["is_mcq"]]
            free_res = [r for r in results if not r["is_mcq"]]
            overall  = acc(results)
            print(f"  MCQ       : {sum(r['correct'] for r in mcq_res):3d} / {len(mcq_res):3d}  ({acc(mcq_res):.1f}%)")
            print(f"  Free-form : {sum(r['correct'] for r in free_res):3d} / {len(free_res):3d}  ({acc(free_res):.1f}%)")
            print(f"  Overall   : {sum(r['correct'] for r in results):3d} / {len(results):3d}  ({overall:.1f}%)\n")
            summary.append((budget, overall, acc(mcq_res), acc(free_res)))
        else:
            results = [
                {"id": item.get("id"), "is_mcq": bool(item.get("options")), "response": resp}
                for item, resp in zip(data, responses)
            ]

        out_file = out_dir / f"results_tokens{budget}.jsonl"
        with open(out_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  Saved → {out_file}")

    # ── 6. Comparison table ────────────────────────────────────────────────────
    if summary:
        print(f"\n{'='*60}")
        print("TOKEN BUDGET COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Budget':>8}  {'Overall':>9}  {'MCQ':>9}  {'Free':>9}")
        print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}")
        for budget, ov, mq, fr in summary:
            print(f"  {budget:>8}  {ov:>8.1f}%  {mq:>8.1f}%  {fr:>8.1f}%")
        print(f"{'='*60}")
        plot_summary(summary, out_dir, N_QUESTIONS)


if __name__ == "__main__":
    main()
