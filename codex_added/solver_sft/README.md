# Qwen Solver SFT

This directory trains a separate solver LoRA for `Qwen/Qwen3-4B-Thinking-2507`.
It does not replace or merge the existing answer finalizer.

Pipeline:

```text
base Qwen + solver_sft_lora, 8k reasoning
  -> existing answer finalizer LoRA
  -> optional MCQ rerank later
```

Default adapter paths:

- Solver adapter: `codex_added/solver_sft/adapters/solver_sft_lora`
- Existing finalizer adapter alias: `codex_added/solver_sft/adapters/answer_finalizer_lora` -> `codex_added/models/qwen3_answer_format_lora`

Do not merge these adapters unless that is explicitly requested.

## Install

From the repo root:

```bash
export REPO_ROOT="$PWD"
export PIP_CACHE_DIR="$REPO_ROOT/.pip-cache"
export UV_CACHE_DIR="$REPO_ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$REPO_ROOT/.uv-python"
export HF_HOME="$REPO_ROOT/.hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_ENABLE_HF_TRANSFER=1
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export TMPDIR="$REPO_ROOT/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

python3 -m pip install --prefix "$REPO_ROOT/.uv-local" uv
export PATH="$REPO_ROOT/.uv-local/bin:$PATH"
uv venv "$REPO_ROOT/.venv-solver-sft" --python 3.11 --seed
source "$REPO_ROOT/.venv-solver-sft/bin/activate"
uv pip install --torch-backend=auto \
  "torch" \
  "accelerate==1.13.0" \
  "antlr4-python3-runtime==4.11.1" \
  "bitsandbytes>=0.48.1" \
  "datasets>=2.20.0" \
  "hf_transfer>=0.1.8" \
  "numpy" \
  "peft==0.15.2" \
  "PyYAML" \
  "sympy" \
  "tqdm" \
  "transformers==4.57.3" \
  "trl==0.19.1"
```

## Data Prep

Put correct self-generated traces here, or pass your own path:

```text
codex_added/solver_sft/data/self_generated_correct.jsonl
```

Expected self-trace format:

```json
{"problem":"...","reasoning":"...","answer":"...","judge_correct":true,"source":"qwen_self_sample"}
```

Build the curated mixture:

```bash
python codex_added/solver_sft/prepare_sft_data.py \
  --config codex_added/solver_sft/configs/full_8k_l40s.yaml
```

The defaults target about 50k examples: 20k MetaMathQA, 7.5k MathInstruct,
7.5k OpenMathInstruct-2, up to 10k local correct self traces, and up to 5k
competition-format rows.

## Train

Main two-GPU launch:

```bash
accelerate launch --num_processes 2 codex_added/solver_sft/train_solver_sft.py \
  --model_name Qwen/Qwen3-4B-Thinking-2507 \
  --train_file codex_added/solver_sft/data/train.jsonl \
  --eval_file codex_added/solver_sft/data/eval.jsonl \
  --output_dir codex_added/solver_sft/adapters/solver_sft_lora \
  --max_seq_length 8192 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --bf16 \
  --gradient_checkpointing \
  --use_4bit
```

Equivalent config-driven command:

```bash
accelerate launch --num_processes 2 codex_added/solver_sft/train_solver_sft.py \
  --config codex_added/solver_sft/configs/full_8k_l40s.yaml
```

Resume from the latest checkpoint:

```bash
accelerate launch --num_processes 2 codex_added/solver_sft/train_solver_sft.py \
  --config codex_added/solver_sft/configs/full_8k_l40s.yaml \
  --resume_from_checkpoint last
```

If 8192 OOMs, use the 6k fallback before considering 4k:

```bash
accelerate launch --num_processes 2 codex_added/solver_sft/train_solver_sft.py \
  --config codex_added/solver_sft/configs/fallback_6k_l40s.yaml
```

Training stability fallbacks:

- Lower LR to `5e-5`.
- Use `--num_train_epochs 0.5`.
- Reduce noisy external data counts.
- Increase the self-generated correct trace proportion.

## Evaluate

Run solver generation, then the existing answer finalizer:

```bash
python codex_added/solver_sft/eval_solver.py \
  --config codex_added/solver_sft/configs/full_8k_l40s.yaml \
  --data_file codex_added/job_data/public_stratified_40.jsonl \
  --solver_adapter codex_added/solver_sft/adapters/solver_sft_lora \
  --answer_finalizer_adapter codex_added/solver_sft/adapters/answer_finalizer_lora
```

The report includes overall, MCQ, free-form accuracy, parse failures,
truncation count, and slot-count mismatches.

To evaluate raw solver formatting without the finalizer:

```bash
python codex_added/solver_sft/eval_solver.py \
  --config codex_added/solver_sft/configs/full_8k_l40s.yaml \
  --without_finalizer
```

## Submission

Generate `id,answer` CSV:

```bash
python codex_added/solver_sft/generate_submission.py \
  --config codex_added/solver_sft/configs/full_8k_l40s.yaml \
  --test_file data/private.jsonl \
  --solver_adapter codex_added/solver_sft/adapters/solver_sft_lora \
  --answer_finalizer_adapter codex_added/solver_sft/adapters/answer_finalizer_lora \
  --output_csv codex_added/solver_sft/submissions/solver_sft_submission.csv
```

If the competition uploader expects `id,response`, use:

```bash
python codex_added/solver_sft/generate_submission.py \
  --config codex_added/solver_sft/configs/full_8k_l40s.yaml \
  --answer_column response
```

## Kubernetes

Launch the full data-prep and training job:

```bash
kubectl apply -f codex_added/solver_sft/k8s/solver-sft-l40s-job.yaml
```

The job requests two `NVIDIA-L40S` GPUs in namespace `ai-md`, mounts the
`biomech-ai` PVC, installs runtime dependencies, prepares the filtered dataset,
and launches `accelerate` with two processes.
