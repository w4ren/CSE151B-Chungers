#!/usr/bin/env bash
# Added by Codex: 100-question raw 8k + LoRA finalizer + verifier/repair run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CODEX_ROOT}/.." && pwd)"
ARCHIVE2_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
DATA_PATH="${DATA_PATH:-codex_added/job_data/public_diagnostic_100_offset200_40mcq_40multislot_20hard.jsonl}"
SOURCE_DATA_PATH="${SOURCE_DATA_PATH:-data/public.jsonl}"
OUT_DIR="${OUT_DIR:-codex_added/results/public100_8k_lora_verify_repair_2gpu_${RUN_ID}}"
SMOKE_DIR="${SMOKE_DIR:-codex_added/results/public100_8k_lora_verify_repair_smoke_${RUN_ID}}"
ROW_LIMIT="${ROW_LIMIT:-100}"
ROW_OFFSET="${ROW_OFFSET:-0}"
GPUS="${GPUS:-0,1}"
EXPECTED_GPUS="${EXPECTED_GPUS:-2}"
MIN_GPU_MEM_MIB="${MIN_GPU_MEM_MIB:-22000}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-1000}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
REASON_MAX_MODEL_LEN="${REASON_MAX_MODEL_LEN:-10240}"
REASON_GPU_UTIL="${REASON_GPU_UTIL:-0.90}"
REASON_MAX_NUM_SEQS="${REASON_MAX_NUM_SEQS:-2}"
REASON_MAX_NUM_BATCHED_TOKENS="${REASON_MAX_NUM_BATCHED_TOKENS:-10240}"
REASON_VLLM_BATCH_SIZE="${REASON_VLLM_BATCH_SIZE:-1}"

FINAL_MAX_MODEL_LEN="${FINAL_MAX_MODEL_LEN:-6144}"
FINAL_GPU_UTIL="${FINAL_GPU_UTIL:-0.90}"
FINAL_MAX_NUM_SEQS="${FINAL_MAX_NUM_SEQS:-2}"
FINAL_MAX_NUM_BATCHED_TOKENS="${FINAL_MAX_NUM_BATCHED_TOKENS:-6144}"
FINAL_VLLM_BATCH_SIZE="${FINAL_VLLM_BATCH_SIZE:-8}"
FINAL_MAX_INPUT_TOKENS="${FINAL_MAX_INPUT_TOKENS:-4096}"
FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS:-64}"
FINAL_RETRY_MAX_NEW_TOKENS="${FINAL_RETRY_MAX_NEW_TOKENS:-128}"

VERIFY_MAX_MODEL_LEN="${VERIFY_MAX_MODEL_LEN:-8192}"
VERIFY_GPU_UTIL="${VERIFY_GPU_UTIL:-0.90}"
VERIFY_MAX_NUM_SEQS="${VERIFY_MAX_NUM_SEQS:-1}"
VERIFY_MAX_NUM_BATCHED_TOKENS="${VERIFY_MAX_NUM_BATCHED_TOKENS:-8192}"
VERIFY_VLLM_BATCH_SIZE="${VERIFY_VLLM_BATCH_SIZE:-1}"
VERIFY_MAX_INPUT_TOKENS="${VERIFY_MAX_INPUT_TOKENS:-6144}"
VERIFY_MAX_NEW_TOKENS="${VERIFY_MAX_NEW_TOKENS:-1024}"
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-prefetch}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHINFER}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-none}"
VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-auto}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

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
export VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-${ARCHIVE2_ROOT}/.vllm-rpc}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${REPO_ROOT}/.pip-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${REPO_ROOT}/.uv-python}"

validate_gpu_csv() {
  local gpus_csv="$1"
  local expected_count="${2:-}"
  local gpu_ids=()
  IFS=',' read -r -a gpu_ids <<< "${gpus_csv}"
  if [[ "${#gpu_ids[@]}" -lt 1 ]]; then
    echo "GPUS must contain at least one GPU id." >&2
    exit 1
  fi
  if [[ -n "${expected_count}" && "${#gpu_ids[@]}" -ne "${expected_count}" ]]; then
    echo "GPUS=${gpus_csv} has ${#gpu_ids[@]} id(s), but EXPECTED_GPUS=${expected_count}." >&2
    echo "Set EXPECTED_GPUS to match intentional one-GPU reruns." >&2
    exit 1
  fi
  local seen=","
  local gpu_id
  for gpu_id in "${gpu_ids[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
      echo "Invalid GPU id in GPUS=${gpus_csv}: ${gpu_id}" >&2
      exit 1
    fi
    if [[ "${seen}" == *",${gpu_id},"* ]]; then
      echo "Duplicate GPU id in GPUS=${gpus_csv}. Refusing to stack shards on one GPU." >&2
      exit 1
    fi
    seen+="${gpu_id},"
  done
}

validate_gpu_csv "${GPUS}" "${EXPECTED_GPUS}"

declare -a VLLM_EXTRA_ARGS=()
if [[ "${VLLM_QUANTIZATION}" != "none" && "${VLLM_QUANTIZATION}" != "false" && "${VLLM_QUANTIZATION}" != "no" ]]; then
  VLLM_EXTRA_ARGS+=(--vllm-quantization "${VLLM_QUANTIZATION}" --vllm-load-format "${VLLM_LOAD_FORMAT}")
elif [[ "${VLLM_LOAD_FORMAT}" != "auto" ]]; then
  VLLM_EXTRA_ARGS+=(--vllm-load-format "${VLLM_LOAD_FORMAT}")
fi
if [[ "${ENFORCE_EAGER}" == "1" || "${ENFORCE_EAGER}" == "true" || "${ENFORCE_EAGER}" == "yes" ]]; then
  VLLM_EXTRA_ARGS+=(--enforce-eager)
fi

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/raw" "${OUT_DIR}/finalized" "${OUT_DIR}/verify_repair" \
  "${SMOKE_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" "${VLLM_RPC_BASE_PATH}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}" \
  "${UV_PYTHON_INSTALL_DIR}"

exec > >(tee -a "${OUT_DIR}/public100_full_pipeline_run.log") 2>&1

echo "run_id=${RUN_ID}"
echo "repo_root=${REPO_ROOT}"
echo "data_path=${DATA_PATH}"
echo "row_offset=${ROW_OFFSET}"
echo "row_limit=${ROW_LIMIT}"
echo "gpus=${GPUS}"
echo "out_dir=${OUT_DIR}"
echo "attention_backend=${ATTENTION_BACKEND}"
echo "safetensors_load_strategy=${SAFETENSORS_LOAD_STRATEGY}"
echo "max_gpu_used_mib=${MAX_GPU_USED_MIB}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "reason_vllm_batch_size=${REASON_VLLM_BATCH_SIZE}"
echo "reason_max_num_seqs=${REASON_MAX_NUM_SEQS}"
echo "verify_vllm_batch_size=${VERIFY_VLLM_BATCH_SIZE}"
echo "vllm_quantization=${VLLM_QUANTIZATION}"
echo "vllm_load_format=${VLLM_LOAD_FORMAT}"
echo "enforce_eager=${ENFORCE_EAGER}"
echo "started_at=$(date -u -Iseconds)"

if [[ ! -f "${DATA_PATH}" ]]; then
  "${PYTHON_BIN:-python}" codex_added/scripts/30_build_public_diagnostic_slice.py \
    --data "${SOURCE_DATA_PATH}" \
    --output "${DATA_PATH}"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi
  gpu_mem_report="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
  echo "GPU memory report:"
  echo "${gpu_mem_report}"
  while IFS=, read -r gpu_name gpu_mem; do
    mem_num="$(printf '%s' "${gpu_mem}" | tr -cd '0-9')"
    if [[ -z "${mem_num}" || "${mem_num}" -lt "${MIN_GPU_MEM_MIB}" ]]; then
      echo "GPU ${gpu_name} has ${mem_num:-unknown} MiB; this 8k vLLM job requires at least ${MIN_GPU_MEM_MIB} MiB per GPU." >&2
      exit 1
    fi
  done <<< "${gpu_mem_report}"
  IFS=',' read -r -a selected_gpu_ids <<< "${GPUS}"
  for gpu_id in "${selected_gpu_ids[@]}"; do
    used_mib="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -cd '0-9')"
    if [[ -z "${used_mib}" ]]; then
      echo "Could not read used memory for GPU ${gpu_id}." >&2
      exit 1
    fi
    if [[ "${used_mib}" -gt "${MAX_GPU_USED_MIB}" ]]; then
      echo "GPU ${gpu_id} already has ${used_mib} MiB in use; refusing to start. Override MAX_GPU_USED_MIB if intentional." >&2
      exit 1
    fi
  done
else
  echo "nvidia-smi is not available in this container." >&2
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
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing local model snapshot: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "Missing finalizer LoRA adapter: ${ADAPTER_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" -c "import accelerate, sys, torch, vllm, transformers, peft; print('python', sys.version.replace('\n', ' ')); print('torch', torch.__version__); print('vllm', vllm.__version__); print('transformers', transformers.__version__); print('peft', peft.__version__); print('accelerate', accelerate.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count()); assert torch.cuda.is_available(); assert torch.cuda.device_count() >= int('${EXPECTED_GPUS}')"

run_pipeline() {
  local data_path="$1"
  local out_dir="$2"
  local limit="$3"
  local offset="$4"
  local gpus="$5"

  mkdir -p "${out_dir}/logs" "${out_dir}/raw" "${out_dir}/finalized" "${out_dir}/verify_repair"
  IFS=',' read -r -a gpu_ids <<< "${gpus}"
  local shard_count="${#gpu_ids[@]}"
  local base_rows=$((limit / shard_count))
  local extra_rows=$((limit % shard_count))
  local current_offset="${offset}"
  local pids=()
  local raw_shards=()
  local final_shards=()
  local repair_shards=()

  validate_gpu_csv "${gpus}"

  wait_for_stage() {
    local stage="$1"
    shift
    local failed=0
    local pid
    for pid in "$@"; do
      if ! wait "${pid}"; then
        failed=1
      fi
    done
    if [[ "${failed}" -ne 0 ]]; then
      echo "${stage} failed. Check ${out_dir}/logs." >&2
      exit 1
    fi
  }

  echo "Starting raw 8k stage: data=${data_path}, limit=${limit}, offset=${offset}, gpus=${gpus}"
  for shard_idx in "${!gpu_ids[@]}"; do
    local shard_limit="${base_rows}"
    if [[ "${shard_idx}" -lt "${extra_rows}" ]]; then
      shard_limit=$((shard_limit + 1))
    fi
    local shard_name
    shard_name="$(printf "%02d" "${shard_idx}")"
    local raw_path="${out_dir}/raw/shard_${shard_name}.jsonl"
    raw_shards+=("${raw_path}")
    if [[ "${shard_limit}" -gt 0 ]]; then
      (
        export CUDA_VISIBLE_DEVICES="${gpu_ids[${shard_idx}]}"
        "${PYTHON_BIN}" codex_added/scripts/10_run_prompt_sweep.py \
          --input "${data_path}" \
          --output "${raw_path}" \
          --variants final_box_only \
          --num-samples 1 \
          --do-sample \
          --max-new-tokens "${MAX_NEW_TOKENS}" \
          --temperature 0.6 \
          --top-p 0.95 \
          --top-k 20 \
          --answer-key-mode strict \
          --backend vllm \
          --model-id "${MODEL_PATH}" \
          --quantization none \
          --dtype float16 \
          --gpu-memory-utilization "${REASON_GPU_UTIL}" \
          --max-model-len "${REASON_MAX_MODEL_LEN}" \
          --max-num-seqs "${REASON_MAX_NUM_SEQS}" \
          --max-num-batched-tokens "${REASON_MAX_NUM_BATCHED_TOKENS}" \
          --vllm-batch-size "${REASON_VLLM_BATCH_SIZE}" \
          "${VLLM_EXTRA_ARGS[@]}" \
          --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}" \
          --attention-backend "${ATTENTION_BACKEND}" \
          --offset "${current_offset}" \
          --limit "${shard_limit}"
      ) > "${out_dir}/logs/reason_8k_shard_${shard_name}.log" 2>&1 &
      pids+=("$!")
      echo "Started raw shard ${shard_name}: gpu=${gpu_ids[${shard_idx}]}, offset=${current_offset}, limit=${shard_limit}, pid=${pids[-1]}"
    fi
    current_offset=$((current_offset + shard_limit))
  done
  wait_for_stage "raw 8k reasoner" "${pids[@]}"
  pids=()

  "${PYTHON_BIN}" codex_added/scripts/21_merge_shard_predictions.py \
    --data "${data_path}" \
    --offset "${offset}" \
    --limit "${limit}" \
    --predictions "${raw_shards[@]}" \
    --output "${out_dir}/raw_8k_merged.jsonl"

  echo "Starting LoRA finalizer stage"
  for shard_idx in "${!gpu_ids[@]}"; do
    local shard_name
    shard_name="$(printf "%02d" "${shard_idx}")"
    local raw_path="${out_dir}/raw/shard_${shard_name}.jsonl"
    local final_path="${out_dir}/finalized/shard_${shard_name}.jsonl"
    final_shards+=("${final_path}")
    (
      export CUDA_VISIBLE_DEVICES="${gpu_ids[${shard_idx}]}"
      "${PYTHON_BIN}" codex_added/scripts/16_finalize_with_qwen.py \
        --data "${data_path}" \
        --responses "${raw_path}" \
        --output "${final_path}" \
        --model-id "${MODEL_PATH}" \
        --adapter-dir "${ADAPTER_DIR}" \
        --backend vllm \
        --dtype float16 \
        --gpu-memory-utilization "${FINAL_GPU_UTIL}" \
        --max-model-len "${FINAL_MAX_MODEL_LEN}" \
        --max-num-seqs "${FINAL_MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${FINAL_MAX_NUM_BATCHED_TOKENS}" \
        --vllm-batch-size "${FINAL_VLLM_BATCH_SIZE}" \
        "${VLLM_EXTRA_ARGS[@]}" \
        --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}" \
        --attention-backend "${ATTENTION_BACKEND}" \
        --max-input-tokens "${FINAL_MAX_INPUT_TOKENS}" \
        --max-new-tokens "${FINAL_MAX_NEW_TOKENS}" \
        --temperature 0.1 \
        --top-p 0.9 \
        --retry-bad-format \
        --retry-max-new-tokens "${FINAL_RETRY_MAX_NEW_TOKENS}" \
        --score
    ) > "${out_dir}/logs/finalize_lora_shard_${shard_name}.log" 2>&1 &
    pids+=("$!")
    echo "Started finalizer shard ${shard_name}: gpu=${gpu_ids[${shard_idx}]}, pid=${pids[-1]}"
  done
  wait_for_stage "LoRA finalizer" "${pids[@]}"
  pids=()

  "${PYTHON_BIN}" codex_added/scripts/21_merge_shard_predictions.py \
    --data "${data_path}" \
    --offset "${offset}" \
    --limit "${limit}" \
    --predictions "${final_shards[@]}" \
    --output "${out_dir}/finalized_8k_merged.jsonl"

  echo "Starting Qwen verifier/repair stage"
  for shard_idx in "${!gpu_ids[@]}"; do
    local shard_name
    shard_name="$(printf "%02d" "${shard_idx}")"
    local raw_path="${out_dir}/raw/shard_${shard_name}.jsonl"
    local final_path="${out_dir}/finalized/shard_${shard_name}.jsonl"
    local repair_path="${out_dir}/verify_repair/shard_${shard_name}.jsonl"
    repair_shards+=("${repair_path}")
    (
      export CUDA_VISIBLE_DEVICES="${gpu_ids[${shard_idx}]}"
      "${PYTHON_BIN}" codex_added/scripts/31_verify_repair_with_qwen.py \
        --data "${data_path}" \
        --raw-responses "${raw_path}" \
        --finalized-responses "${final_path}" \
        --output "${repair_path}" \
        --model-id "${MODEL_PATH}" \
        --backend vllm \
        --dtype float16 \
        --gpu-memory-utilization "${VERIFY_GPU_UTIL}" \
        --max-model-len "${VERIFY_MAX_MODEL_LEN}" \
        --max-num-seqs "${VERIFY_MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${VERIFY_MAX_NUM_BATCHED_TOKENS}" \
        --vllm-batch-size "${VERIFY_VLLM_BATCH_SIZE}" \
        "${VLLM_EXTRA_ARGS[@]}" \
        --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}" \
        --attention-backend "${ATTENTION_BACKEND}" \
        --max-input-tokens "${VERIFY_MAX_INPUT_TOKENS}" \
        --max-new-tokens "${VERIFY_MAX_NEW_TOKENS}" \
        --temperature 0.1 \
        --top-p 0.9 \
        --score
    ) > "${out_dir}/logs/verify_repair_shard_${shard_name}.log" 2>&1 &
    pids+=("$!")
    echo "Started verifier/repair shard ${shard_name}: gpu=${gpu_ids[${shard_idx}]}, pid=${pids[-1]}"
  done
  wait_for_stage "Qwen verifier/repair" "${pids[@]}"

  "${PYTHON_BIN}" codex_added/scripts/21_merge_shard_predictions.py \
    --data "${data_path}" \
    --offset "${offset}" \
    --limit "${limit}" \
    --predictions "${repair_shards[@]}" \
    --output "${out_dir}/verify_repair_merged.jsonl"

  "${PYTHON_BIN}" codex_added/scripts/32_score_stage_progression.py \
    --data "${data_path}" \
    --stage raw_8k "${out_dir}/raw_8k_merged.jsonl" \
    --stage lora_finalizer "${out_dir}/finalized_8k_merged.jsonl" \
    --stage verify_repair "${out_dir}/verify_repair_merged.jsonl" \
    --output-json "${out_dir}/stage_progression.json" \
    --output-md "${out_dir}/stage_progression.md" \
    --scored-dir "${out_dir}/scored"
}

if [[ "${RUN_SMOKE}" == "1" ]]; then
  echo "Starting one-question smoke test at $(date -u -Iseconds)"
  run_pipeline "${DATA_PATH}" "${SMOKE_DIR}" 1 0 "$(printf '%s' "${GPUS}" | cut -d, -f1)"
  echo "Smoke test finished at $(date -u -Iseconds)"
fi

run_pipeline "${DATA_PATH}" "${OUT_DIR}" "${ROW_LIMIT}" "${ROW_OFFSET}" "${GPUS}"

{
  echo "run_id=${RUN_ID}"
  echo "data_path=${DATA_PATH}"
  echo "row_offset=${ROW_OFFSET}"
  echo "row_limit=${ROW_LIMIT}"
  echo "gpus=${GPUS}"
  echo "model_path=${MODEL_PATH}"
  echo "adapter_dir=${ADAPTER_DIR}"
  echo "python_bin=${PYTHON_BIN}"
  echo "attention_backend=${ATTENTION_BACKEND}"
  echo "safetensors_load_strategy=${SAFETENSORS_LOAD_STRATEGY}"
  echo "max_gpu_used_mib=${MAX_GPU_USED_MIB}"
  echo "max_new_tokens=${MAX_NEW_TOKENS}"
  echo "reason_vllm_batch_size=${REASON_VLLM_BATCH_SIZE}"
  echo "reason_max_num_seqs=${REASON_MAX_NUM_SEQS}"
  echo "verify_vllm_batch_size=${VERIFY_VLLM_BATCH_SIZE}"
  echo "vllm_quantization=${VLLM_QUANTIZATION}"
  echo "vllm_load_format=${VLLM_LOAD_FORMAT}"
  echo "enforce_eager=${ENFORCE_EAGER}"
  echo "raw_output=${OUT_DIR}/raw_8k_merged.jsonl"
  echo "finalized_output=${OUT_DIR}/finalized_8k_merged.jsonl"
  echo "verify_repair_output=${OUT_DIR}/verify_repair_merged.jsonl"
  echo "stage_progression=${OUT_DIR}/stage_progression.md"
  echo "completed_at=$(date -u -Iseconds)"
} > "${OUT_DIR}/RUN_INFO.txt"

echo "completed_at=$(date -u -Iseconds)"
echo "Raw output: ${OUT_DIR}/raw_8k_merged.jsonl"
echo "Finalizer output: ${OUT_DIR}/finalized_8k_merged.jsonl"
echo "Verifier/repair output: ${OUT_DIR}/verify_repair_merged.jsonl"
echo "Progression summary: ${OUT_DIR}/stage_progression.md"
