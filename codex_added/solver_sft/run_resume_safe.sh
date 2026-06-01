#!/usr/bin/env bash
set -euxo pipefail

export REPO_ROOT="${REPO_ROOT:-$PWD}"
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
export PYTHONUNBUFFERED=1

source "$REPO_ROOT/.venv-solver-sft/bin/activate"

resume_args=()
if find codex_added/solver_sft/adapters/solver_sft_lora -maxdepth 1 -type d -name "checkpoint-*" | grep -q .; then
  resume_args+=(--resume_from_checkpoint last)
fi

exec accelerate launch --num_processes 2 \
  codex_added/solver_sft/train_solver_sft.py \
  --config codex_added/solver_sft/configs/full_8k_l40s.yaml \
  --no-tf32 \
  --optim adamw_torch \
  --save_steps 100 \
  --keep_last_n_checkpoints 5 \
  "${resume_args[@]}"
