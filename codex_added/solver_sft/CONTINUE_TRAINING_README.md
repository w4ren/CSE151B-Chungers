# Continue Solver SFT Training

This file is the handoff note for continuing the direct in-pod QLoRA SFT run. Keep all commands scoped to this repo:

```bash
cd /home/mnt/BiomechAI/Wenhao/archive2/CSE151B-Chungers-wenhao-lfs
```

## Current Run

- This is not a Kubernetes Job. It is a direct `accelerate launch` process in the current pod.
- Model: `Qwen/Qwen3-4B-Thinking-2507`
- Training data already exists:
  - `codex_added/solver_sft/data/train.jsonl`
  - `codex_added/solver_sft/data/eval.jsonl`
- Clean output directory:
  - `codex_added/solver_sft/adapters/solver_sft_lora`
- Checkpoint cadence:
  - `--save_steps 100`
  - `--keep_last_n_checkpoints 5`
- Optimizer:
  - `adamw_torch`
  - Do not switch back to `paged_adamw_8bit` for resumability. A bitsandbytes optimizer resume failed with `Error invalid argument at line 689 in file /src/csrc/pythonInterface.cpp`.
- TF32 is disabled with `--no-tf32`; leaving `tf32: true` from the YAML caused `transformers` to reject the device capability.
- An older interrupted bitsandbytes run was preserved here:
  - `codex_added/solver_sft/adapters/solver_sft_lora_bnb_interrupted_20260528T0156`

## Check Status

```bash
pgrep -af 'train_solver_sft.py|accelerate launch' || true
find codex_added/solver_sft/adapters/solver_sft_lora -maxdepth 2 -type d -name 'checkpoint-*' -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort
```

If no `checkpoint-*` directory exists yet, the current clean run has not reached step 100. If the pod dies before the first checkpoint, restart from step 0 with the runner below.

## Continue Or Restart

Use the runner script. It automatically resumes from the latest checkpoint if one exists; otherwise it starts from scratch.

```bash
./codex_added/solver_sft/run_resume_safe.sh
```

Equivalent explicit resume command, if checkpoints exist:

```bash
set -euo pipefail
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
export TRITON_CACHE_DIR="$REPO_ROOT/.triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.torchinductor-cache"
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$REPO_ROOT/.venv-solver-sft/bin/activate"

accelerate launch --num_processes 2 \
  codex_added/solver_sft/train_solver_sft.py \
  --config codex_added/solver_sft/configs/full_8k_l40s.yaml \
  --no-tf32 \
  --optim adamw_torch \
  --save_steps 100 \
  --keep_last_n_checkpoints 5 \
  --resume_from_checkpoint last
```

## Prompt For Next Codex Session

```text
Stay strictly inside /home/mnt/BiomechAI/Wenhao/archive2. Continue the direct in-pod 2-GPU Qwen solver SFT training, not as a Kubernetes Job. Work in /home/mnt/BiomechAI/Wenhao/archive2/CSE151B-Chungers-wenhao-lfs. Read codex_added/solver_sft/CONTINUE_TRAINING_README.md first. Check whether training is still running with pgrep. Check latest checkpoints in codex_added/solver_sft/adapters/solver_sft_lora. If training is not running, resume using ./codex_added/solver_sft/run_resume_safe.sh. Keep --no-tf32, --optim adamw_torch, --save_steps 100, and --keep_last_n_checkpoints 5. Do not use the old bitsandbytes interrupted output directory except for reference. Do not run kubectl or create a Kubernetes Job.
```
