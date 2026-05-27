"""
CSE 151B Spring 2026 — Fine-tuning script
QLoRA SFT on public math datasets → better math reasoning for Qwen3-4B-Thinking

Recommended datasets (choose via --dataset):
  numina      : AI-MO/NuminaMath-CoT        (860k competition problems + CoT)
  openmath    : nvidia/OpenMathInstruct-2   (filtered 1M high-quality)
  openthoughts: open-thoughts/OpenThoughts-114k  (think-format, matches Qwen3)
  math        : lighteval/MATH              (12.5k competition problems)
  gsm8k       : openai/gsm8k               (8.5k grade school, numerical)

Usage:
    python train.py --dataset numina --output ./checkpoints/numina-qlora
    python train.py --dataset openthoughts --samples 50000 --epochs 1

Requirements:
    pip install transformers trl peft datasets bitsandbytes accelerate
    (Optional for speed) pip install unsloth  ← 2x faster, less memory
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen3-4B-Thinking-2507"
MAX_LENGTH = 8192   # max tokens per sample (prompt + completion)

DATASET_CONFIGS = {
    "numina": {
        "path": "AI-MO/NuminaMath-CoT",
        "split": "train",
        "question_col": "problem",
        "solution_col": "solution",   # contains full CoT + \boxed{} answer
        "has_thinking": False,         # solution is CoT but not <think> wrapped
    },
    "openmath": {
        "path": "nvidia/OpenMathInstruct-2",
        "split": "train_1M",           # or "train_5M" for more data
        "question_col": "problem",
        "solution_col": "generated_solution",
        "has_thinking": False,
    },
    "openthoughts": {
        "path": "open-thoughts/OpenThoughts-114k",
        "split": "train",
        "question_col": "problem",
        "solution_col": "solution",    # already <think>…</think>\n<answer>
        "has_thinking": True,          # preserves thinking format exactly
    },
    "math": {
        "path": "lighteval/MATH",
        "split": "train",
        "question_col": "problem",
        "solution_col": "solution",
        "has_thinking": False,
    },
    "gsm8k": {
        "path": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "question_col": "question",
        "solution_col": "answer",
        "has_thinking": False,
    },
}

# ── Prompt helpers ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step by step, "
    "showing your reasoning clearly. Put your final answer inside \\boxed{}."
)


def format_sample_sft(example: dict, tokenizer, cfg: dict) -> dict:
    """
    Convert a raw dataset row into a chat-formatted training string.
    For thinking models, the assistant turn wraps reasoning in <think>…</think>.
    """
    question = example.get(cfg["question_col"], "").strip()
    solution = example.get(cfg["solution_col"], "").strip()

    if not question or not solution:
        return {"text": ""}

    if cfg["has_thinking"]:
        # Dataset already has <think>…</think> format — use as-is
        assistant_content = solution
    else:
        # Wrap the solution in thinking tags so the model learns to think
        assistant_content = f"<think>\n{solution}\n</think>"
        # Try to find \boxed{} answer and append it after the think block
        import re
        boxed_matches = re.findall(r"\\boxed\{[^}]+\}", solution)
        if boxed_matches:
            last_box = boxed_matches[-1]
            assistant_content += f"\n\nThe answer is {last_box}."

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": question},
        {"role": "assistant", "content": assistant_content},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ── QLoRA setup ───────────────────────────────────────────────────────────────
def get_bnb_config(bits: int = 4) -> BitsAndBytesConfig:
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def get_lora_config(rank: int = 64) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen3-4B-Thinking on math data")
    parser.add_argument("--dataset",    default="numina",    choices=list(DATASET_CONFIGS.keys()),
                        help="Which public math dataset to train on")
    parser.add_argument("--output",     default="./checkpoints/math-qlora",
                        help="Output directory for checkpoints")
    parser.add_argument("--samples",    default=50000, type=int,
                        help="Max training samples (0 = use all)")
    parser.add_argument("--epochs",     default=1, type=int)
    parser.add_argument("--batch-size", default=2, type=int,
                        help="Per-device batch size (use 1-2 with QLoRA on 40GB GPU)")
    parser.add_argument("--grad-accum", default=8, type=int,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--lr",         default=2e-4, type=float)
    parser.add_argument("--lora-rank",  default=64, type=int,
                        help="LoRA rank — higher = more parameters, potentially better results")
    parser.add_argument("--bits",       default=4, type=int, choices=[4, 8],
                        help="Quantization bits (4=QLoRA, 8=LoRA with INT8)")
    parser.add_argument("--gpu",        default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--mix-gsm8k",  action="store_true",
                        help="Also mix in GSM8K for numerical reasoning diversity")
    parser.add_argument("--base-model", default=MODEL_ID,
                        help="HuggingFace model ID (or local path to a prior checkpoint)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = DATASET_CONFIGS[args.dataset]
    print(f"\n{'='*60}")
    print(f"Dataset  : {cfg['path']}")
    print(f"Model    : {args.base_model}")
    print(f"Output   : {args.output}")
    print(f"LoRA rank: {args.lora_rank}  |  Quant: {args.bits}-bit")
    print(f"{'='*60}\n")

    # ── 1. Load tokenizer ──────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # important for SFT

    # ── 2. Load dataset ────────────────────────────────────────────────────────
    print(f"Loading dataset {cfg['path']}...")
    load_kwargs = dict(split=cfg["split"], trust_remote_code=True)
    if "config" in cfg:
        load_kwargs["name"] = cfg["config"]

    raw_ds = load_dataset(cfg["path"], **load_kwargs)
    if args.samples > 0:
        raw_ds = raw_ds.shuffle(seed=42).select(range(min(args.samples, len(raw_ds))))

    # Optional: mix in GSM8K
    if args.mix_gsm8k and args.dataset != "gsm8k":
        print("Mixing in GSM8K...")
        gsm_cfg = DATASET_CONFIGS["gsm8k"]
        gsm_ds = load_dataset(gsm_cfg["path"], gsm_cfg["config"], split=gsm_cfg["split"])
        # Format GSM8K separately then merge
        gsm_ds = gsm_ds.map(
            lambda ex: format_sample_sft(ex, tokenizer, gsm_cfg),
            remove_columns=gsm_ds.column_names,
            num_proc=4,
        ).filter(lambda x: len(x["text"]) > 50)
        raw_formatted = raw_ds.map(
            lambda ex: format_sample_sft(ex, tokenizer, cfg),
            remove_columns=raw_ds.column_names,
            num_proc=4,
        ).filter(lambda x: len(x["text"]) > 50)
        dataset = concatenate_datasets([raw_formatted, gsm_ds]).shuffle(seed=42)
    else:
        # Format the main dataset
        dataset = raw_ds.map(
            lambda ex: format_sample_sft(ex, tokenizer, cfg),
            remove_columns=raw_ds.column_names,
            num_proc=4,
        ).filter(lambda x: len(x["text"]) > 50)

    print(f"Training on {len(dataset):,} samples")

    # ── 3. Load model with QLoRA ───────────────────────────────────────────────
    print("Loading model with QLoRA...")
    bnb_config = get_bnb_config(args.bits)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    lora_cfg = get_lora_config(args.lora_rank)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── 4. Training arguments ──────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",   # memory-efficient optimizer for QLoRA
        report_to="none",           # change to "wandb" if you use W&B
        dataloader_num_workers=2,
        group_by_length=True,       # speeds up training by reducing padding
        max_grad_norm=1.0,
        seed=42,
    )

    # ── 5. Trainer ─────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=MAX_LENGTH,
        packing=True,   # pack multiple short samples into one context window → faster
    )

    # ── 6. Train ───────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── 7. Save LoRA adapter ───────────────────────────────────────────────────
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nLoRA adapter saved to {final_path}")
    print("To run inference with this adapter, pass --base-model to baseline.py (see run_finetuned.py)")

    # Save training metadata
    meta = {
        "base_model": args.base_model,
        "dataset": cfg["path"],
        "samples_trained": len(dataset),
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "bits": args.bits,
        "lr": args.lr,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Training metadata saved to {output_dir / 'training_meta.json'}")


if __name__ == "__main__":
    main()
