# CSE 151B SP26 — Chungers

Fine-tuning and inference pipeline for the CSE 151B Spring 2026 Math Reasoning Competition.  
Base model: **Qwen3-4B-Thinking-2507** — fine-tuned with QLoRA on public math datasets.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/w4ren/CSE151B-Chungers.git
cd CSE151B-Chungers

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install transformers trl peft datasets bitsandbytes accelerate vllm tqdm
```

> **Optional (2× faster training):** `pip install unsloth`

Place the competition data at `data/public.jsonl` (copy from the starter repo).

---

## `train.py` — Fine-tune on public math datasets

Fine-tunes the base model using **QLoRA (4-bit)** + **LoRA adapters** via SFT.  
The adapter is saved to `./checkpoints/<name>/final/` and can be loaded directly by `run_finetuned.py`.

### Supported datasets

| Key | Dataset | Size | Notes |
|---|---|---|---|
| `numina` | `AI-MO/NuminaMath-CoT` | 860k | Competition problems + chain-of-thought |
| `openmath` | `nvidia/OpenMathInstruct-2` | 1M | Diverse math, high quality |
| `openthoughts` | `open-thoughts/OpenThoughts-114k` | 114k | Already in `<think>…</think>` format — best match for Qwen3-Thinking |
| `math` | `lighteval/MATH` | 12.5k | AMC/AIME-level competition problems |
| `gsm8k` | `openai/gsm8k` | 8.5k | Grade-school word problems |

### Usage

```bash
# Basic — train on 50k NuminaMath samples for 1 epoch
python train.py --dataset numina

# Use OpenThoughts (best for thinking-model fine-tuning)
python train.py --dataset openthoughts --epochs 2

# Train on all NuminaMath + mix in GSM8K for numerical diversity
python train.py --dataset numina --mix-gsm8k --samples 100000

# Custom output path and LoRA rank
python train.py --dataset numina --output ./checkpoints/numina-r32 --lora-rank 32

# Use all data (no sample cap)
python train.py --dataset openmath --samples 0 --epochs 1
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `numina` | Which dataset to train on (see table above) |
| `--output` | `./checkpoints/math-qlora` | Where to save checkpoints and final adapter |
| `--samples` | `50000` | Max training samples (`0` = use full dataset) |
| `--epochs` | `1` | Number of training epochs |
| `--batch-size` | `2` | Per-device batch size |
| `--grad-accum` | `8` | Gradient accumulation steps (effective batch = `batch-size × grad-accum`) |
| `--lr` | `2e-4` | Learning rate |
| `--lora-rank` | `64` | LoRA rank — higher captures more, uses more memory |
| `--bits` | `4` | Quantization: `4` = QLoRA (NF4), `8` = INT8 LoRA |
| `--gpu` | `0` | `CUDA_VISIBLE_DEVICES` value |
| `--mix-gsm8k` | off | Also mix in GSM8K for numerical diversity |
| `--base-model` | Qwen3-4B-Thinking-2507 | HF model ID or path to a prior checkpoint |

### Output structure

```
checkpoints/
└── numina-qlora/
    ├── checkpoint-200/        ← intermediate saves
    ├── checkpoint-400/
    ├── final/                 ← use this path with run_finetuned.py
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   └── tokenizer files
    └── training_meta.json     ← dataset/hyperparameter record
```

---

## `run_finetuned.py` — Inference with LoRA adapter

Loads the base Qwen3-4B-Thinking model and optionally applies a LoRA adapter, then runs inference on the competition dataset.

### Usage

```bash
# Run with fine-tuned adapter (recommended)
python run_finetuned.py --adapter ./checkpoints/numina-qlora/final

# Specify custom data and output paths
python run_finetuned.py \
  --adapter ./checkpoints/numina-qlora/final \
  --data data/public.jsonl \
  --output results/finetuned_results.jsonl

# Run on private test set (skip scoring)
python run_finetuned.py \
  --adapter ./checkpoints/numina-qlora/final \
  --data data/private.jsonl \
  --output results/submission.jsonl \
  --no-eval

# Run base model only (no adapter) — equivalent to baseline.py
python run_finetuned.py --data data/public.jsonl
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--adapter` | `None` | Path to LoRA adapter directory (output of `train.py`) |
| `--base-model` | Qwen3-4B-Thinking-2507 | Base model HF ID or local path |
| `--data` | `data/public.jsonl` | Input `.jsonl` file |
| `--output` | `results/finetuned_results.jsonl` | Output `.jsonl` file |
| `--gpu` | `0` | `CUDA_VISIBLE_DEVICES` value |
| `--max-tokens` | `8192` | Max new tokens per response |
| `--limit` | `None` | Only process first N items (for quick testing) |
| `--no-eval` | off | Skip accuracy scoring (use for private test set) |

### Output format

Each line of the output `.jsonl` contains:

```json
{
  "id": 42,
  "is_mcq": false,
  "gold": ["5/8"],
  "response": "<think>\n...\n</think>\n\nThe answer is \\boxed{5/8}.",
  "correct": true
}
```

---

## End-to-end example

```bash
# 1. Fine-tune on OpenThoughts (already in <think> format)
python train.py \
  --dataset openthoughts \
  --epochs 2 \
  --lora-rank 64 \
  --output ./checkpoints/openthoughts-r64

# 2. Evaluate on public set
python run_finetuned.py \
  --adapter ./checkpoints/openthoughts-r64/final \
  --output results/openthoughts_eval.jsonl

# 3. Generate submission on private set
python run_finetuned.py \
  --adapter ./checkpoints/openthoughts-r64/final \
  --data data/private.jsonl \
  --output results/submission.jsonl \
  --no-eval
```

---

## GPU requirements

| Config | VRAM needed |
|---|---|
| 4-bit QLoRA, rank 64 (default) | ~18 GB |
| 4-bit QLoRA, rank 32 | ~14 GB |
| 8-bit LoRA, rank 64 | ~28 GB |

Inference with a loaded adapter requires ~12 GB.
