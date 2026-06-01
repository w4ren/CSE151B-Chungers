#!/usr/bin/env bash
# Added by Codex: Kubernetes-friendly 200-question 8k vLLM run with setup checks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
DATA_PATH="${DATA_PATH:-data/public.jsonl}"
ROW_LIMIT="${ROW_LIMIT:-200}"
ROW_OFFSET="${ROW_OFFSET:-0}"
GPUS="${GPUS:-0,1,2,3}"
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
MIN_GPU_MEM_MIB="${MIN_GPU_MEM_MIB:-22000}"
OUT_DIR="${OUT_DIR:-codex_added/results/k8s_public200_8k_finalized_4gpu_${RUN_ID}}"
SMOKE_DIR="${SMOKE_DIR:-codex_added/results/k8s_public200_8k_smoke_${RUN_ID}}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/.hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea}"
ADAPTER_DIR="${ADAPTER_DIR:-codex_added/models/qwen3_answer_format_lora}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-vllm/bin/python}"
RUN_SMOKE="${RUN_SMOKE:-1}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export TMPDIR="${TMPDIR:-${REPO_ROOT}/.tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${REPO_ROOT}/.triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${REPO_ROOT}/.torchinductor-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${REPO_ROOT}/.vllm-cache}"
export VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-$(cd "${REPO_ROOT}/.." && pwd)/.vllm-rpc}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${REPO_ROOT}/.pip-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${REPO_ROOT}/.uv-python}"

mkdir -p "${OUT_DIR}" "${SMOKE_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${VLLM_RPC_BASE_PATH}" \
  "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

exec > >(tee -a "${OUT_DIR}/k8s_public200_run.log") 2>&1

echo "run_id=${RUN_ID}"
echo "repo_root=${REPO_ROOT}"
echo "data_path=${DATA_PATH}"
echo "row_offset=${ROW_OFFSET}"
echo "row_limit=${ROW_LIMIT}"
echo "gpus=${GPUS}"
echo "out_dir=${OUT_DIR}"
echo "started_at=$(date -u -Iseconds)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi
else
  echo "nvidia-smi is not available in this container." >&2
  exit 1
fi

gpu_mem_report="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
echo "GPU memory report:"
echo "${gpu_mem_report}"
while IFS=, read -r gpu_name gpu_mem; do
  mem_num="$(printf '%s' "${gpu_mem}" | tr -cd '0-9')"
  if [[ -z "${mem_num}" || "${mem_num}" -lt "${MIN_GPU_MEM_MIB}" ]]; then
    echo "GPU ${gpu_name} has ${mem_num:-unknown} MiB; this 8k vLLM job requires at least ${MIN_GPU_MEM_MIB} MiB per GPU to avoid OOM." >&2
    exit 1
  fi
done <<< "${gpu_mem_report}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing data file: ${DATA_PATH}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing local model snapshot: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "Missing finalizer LoRA adapter: ${ADAPTER_DIR}" >&2
  exit 1
fi

check_python_env() {
  [[ -x "${PYTHON_BIN}" ]] || return 1
  "${PYTHON_BIN}" -c "import accelerate, peft, torch, transformers, vllm; assert vllm.__version__ == '0.21.0', vllm.__version__; assert transformers.__version__ == '4.57.3', transformers.__version__; assert peft.__version__ == '0.15.2', peft.__version__; assert accelerate.__version__ == '1.13.0', accelerate.__version__" >/dev/null 2>&1
}

if ! check_python_env; then
  BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
  if ! command -v "${BOOTSTRAP_PYTHON}" >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="python"
  fi
  "${BOOTSTRAP_PYTHON}" -m pip install --prefix "${REPO_ROOT}/.uv-local" uv
  export PATH="${REPO_ROOT}/.uv-local/bin:${PATH}"
  uv venv "${REPO_ROOT}/.venv-vllm" --python 3.11 --seed
  set +u
  source "${REPO_ROOT}/.venv-vllm/bin/activate"
  set -u
  uv pip install --torch-backend=auto -r codex_added/requirements.txt
else
  export PATH="${REPO_ROOT}/.uv-local/bin:${PATH}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment after setup: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" -c "import accelerate, sys, torch, vllm, transformers, peft; print('python', sys.version.replace('\n', ' ')); print('torch', torch.__version__); print('vllm', vllm.__version__); print('transformers', transformers.__version__); print('peft', peft.__version__); print('accelerate', accelerate.__version__); assert vllm.__version__ == '0.21.0', vllm.__version__; assert transformers.__version__ == '4.57.3', transformers.__version__; assert peft.__version__ == '0.15.2', peft.__version__; assert accelerate.__version__ == '1.13.0', accelerate.__version__; print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count()); assert torch.cuda.is_available(); assert torch.cuda.device_count() >= int('${EXPECTED_GPUS}')"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  echo "Starting one-question smoke test at $(date -u -Iseconds)"
  LIMIT=1 \
  OFFSET=0 \
  GPUS=0 \
  PYTHON_BIN="${PYTHON_BIN}" \
  MODEL_PATH="${MODEL_PATH}" \
  ADAPTER_DIR="${ADAPTER_DIR}" \
  MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-64}" \
  REASON_MAX_MODEL_LEN="${SMOKE_REASON_MAX_MODEL_LEN:-${REASON_MAX_MODEL_LEN:-10240}}" \
  REASON_MAX_NUM_SEQS="${SMOKE_REASON_MAX_NUM_SEQS:-1}" \
  REASON_MAX_NUM_BATCHED_TOKENS="${SMOKE_REASON_MAX_NUM_BATCHED_TOKENS:-${REASON_MAX_NUM_BATCHED_TOKENS:-10240}}" \
  REASON_VLLM_BATCH_SIZE="${SMOKE_REASON_VLLM_BATCH_SIZE:-1}" \
  FINAL_MAX_MODEL_LEN="${SMOKE_FINAL_MAX_MODEL_LEN:-${FINAL_MAX_MODEL_LEN:-6144}}" \
  FINAL_MAX_NUM_SEQS="${SMOKE_FINAL_MAX_NUM_SEQS:-1}" \
  FINAL_MAX_NUM_BATCHED_TOKENS="${SMOKE_FINAL_MAX_NUM_BATCHED_TOKENS:-${FINAL_MAX_NUM_BATCHED_TOKENS:-6144}}" \
  FINAL_VLLM_BATCH_SIZE="${SMOKE_FINAL_VLLM_BATCH_SIZE:-1}" \
  FINAL_MAX_NEW_TOKENS="${SMOKE_FINAL_MAX_NEW_TOKENS:-64}" \
  bash codex_added/scripts/27_run_eval40_8k_finalized_2gpu.sh "${DATA_PATH}" "${SMOKE_DIR}"
  echo "Smoke test finished at $(date -u -Iseconds)"
fi

echo "Starting 200-question 8k run at $(date -u -Iseconds)"
LIMIT="${ROW_LIMIT}" \
OFFSET="${ROW_OFFSET}" \
GPUS="${GPUS}" \
PYTHON_BIN="${PYTHON_BIN}" \
MODEL_PATH="${MODEL_PATH}" \
ADAPTER_DIR="${ADAPTER_DIR}" \
bash codex_added/scripts/27_run_eval40_8k_finalized_2gpu.sh "${DATA_PATH}" "${OUT_DIR}"

{
  echo "run_id=${RUN_ID}"
  echo "data_path=${DATA_PATH}"
  echo "row_offset=${ROW_OFFSET}"
  echo "row_limit=${ROW_LIMIT}"
  echo "gpus=${GPUS}"
  echo "model_path=${MODEL_PATH}"
  echo "adapter_dir=${ADAPTER_DIR}"
  echo "python_bin=${PYTHON_BIN}"
  echo "output=${OUT_DIR}/selected_8k_lora_finalized.jsonl"
  echo "completed_at=$(date -u -Iseconds)"
} > "${OUT_DIR}/K8S_RUN_INFO.txt"

echo "completed_at=$(date -u -Iseconds)"
echo "Main output: ${OUT_DIR}/selected_8k_lora_finalized.jsonl"
