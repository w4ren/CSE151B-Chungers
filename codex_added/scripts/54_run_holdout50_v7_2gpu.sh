#!/usr/bin/env bash
# Added by Codex: non-overlapping public-second100 holdout-50 current-best run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CODEX_ROOT}/.." && pwd)"
ARCHIVE2_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
DATA_PATH="${DATA_PATH:-codex_added/job_data/public_second100_holdout50_nonoverlap_v7_eval.jsonl}"
OUT_DIR="${OUT_DIR:-codex_added/results/holdout50_v7_2gpu_${RUN_ID}}"
LIMIT="${LIMIT:-50}"
OFFSET="${OFFSET:-0}"
GPUS="${GPUS:-0,1}"
EXPECTED_GPUS="${EXPECTED_GPUS:-2}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
REASON_MAX_MODEL_LEN="${REASON_MAX_MODEL_LEN:-10240}"
REASON_GPU_UTIL="${REASON_GPU_UTIL:-0.95}"
REASON_MAX_NUM_SEQS="${REASON_MAX_NUM_SEQS:-4}"
REASON_MAX_NUM_BATCHED_TOKENS="${REASON_MAX_NUM_BATCHED_TOKENS:-10240}"
REASON_VLLM_BATCH_SIZE="${REASON_VLLM_BATCH_SIZE:-0}"

FINAL_MAX_MODEL_LEN="${FINAL_MAX_MODEL_LEN:-6144}"
FINAL_GPU_UTIL="${FINAL_GPU_UTIL:-0.90}"
FINAL_MAX_NUM_SEQS="${FINAL_MAX_NUM_SEQS:-4}"
FINAL_MAX_NUM_BATCHED_TOKENS="${FINAL_MAX_NUM_BATCHED_TOKENS:-6144}"
FINAL_VLLM_BATCH_SIZE="${FINAL_VLLM_BATCH_SIZE:-8}"
FINAL_MAX_INPUT_TOKENS="${FINAL_MAX_INPUT_TOKENS:-4096}"
FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS:-64}"
FINAL_RETRY_MAX_NEW_TOKENS="${FINAL_RETRY_MAX_NEW_TOKENS:-128}"

ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHINFER}"
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-prefetch}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/.hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea}"
ADAPTER_DIR="${ADAPTER_DIR:-codex_added/models/qwen3_answer_format_lora}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-vllm/bin/python}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export TMPDIR="${TMPDIR:-${REPO_ROOT}/.tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${REPO_ROOT}/.triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${REPO_ROOT}/.torchinductor-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${REPO_ROOT}/.vllm-cache}"
export VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-${ARCHIVE2_ROOT}/r}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

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
    echo "GPUS=${gpus_csv} has ${#gpu_ids[@]} id(s), expected ${expected_count}." >&2
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
      echo "Duplicate GPU id in GPUS=${gpus_csv}." >&2
      exit 1
    fi
    seen+="${gpu_id},"
  done
}

validate_gpu_csv "${GPUS}" "${EXPECTED_GPUS}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/raw" "${OUT_DIR}/finalized" "${OUT_DIR}/audit" "${OUT_DIR}/scored" \
  "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${VLLM_RPC_BASE_PATH}"

exec > >(tee -a "${OUT_DIR}/holdout50_v7_2gpu_run.log") 2>&1

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing data slice: ${DATA_PATH}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing model snapshot: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "Missing finalizer adapter: ${ADAPTER_DIR}" >&2
  exit 1
fi

echo "run_id=${RUN_ID}"
echo "data_path=${DATA_PATH}"
echo "out_dir=${OUT_DIR}"
echo "offset=${OFFSET}"
echo "limit=${LIMIT}"
echo "gpus=${GPUS}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "reason_max_num_seqs=${REASON_MAX_NUM_SEQS}"
echo "reason_vllm_batch_size=${REASON_VLLM_BATCH_SIZE}"
echo "reason_vllm_batch_size_note=0 means queue the full shard while max_num_seqs controls active concurrency"
echo "attention_backend=${ATTENTION_BACKEND}"
echo "vllm_rpc_base_path=${VLLM_RPC_BASE_PATH}"
echo "started_at=$(date -u -Iseconds)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
else
  echo "nvidia-smi not found." >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import torch, vllm, transformers, peft, accelerate, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("accelerate", accelerate.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
assert torch.cuda.is_available()
PY

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
SHARD_COUNT="${#GPU_IDS[@]}"
BASE_ROWS=$((LIMIT / SHARD_COUNT))
EXTRA_ROWS=$((LIMIT % SHARD_COUNT))
declare -a RAW_SHARDS=()
declare -a FINAL_SHARDS=()
declare -a PIDS=()

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
    echo "${stage} failed. Check ${OUT_DIR}/logs." >&2
    exit 1
  fi
}

echo "Starting raw 8k reasoner at $(date -u -Iseconds)"
current_offset="${OFFSET}"
for shard_idx in "${!GPU_IDS[@]}"; do
  shard_limit="${BASE_ROWS}"
  if [[ "${shard_idx}" -lt "${EXTRA_ROWS}" ]]; then
    shard_limit=$((shard_limit + 1))
  fi
  shard_name="$(printf "%02d" "${shard_idx}")"
  raw_path="${OUT_DIR}/raw/shard_${shard_name}.jsonl"
  RAW_SHARDS+=("${raw_path}")
  if [[ "${shard_limit}" -gt 0 ]]; then
    (
      export CUDA_VISIBLE_DEVICES="${GPU_IDS[${shard_idx}]}"
      "${PYTHON_BIN}" codex_added/scripts/10_run_prompt_sweep.py \
        --input "${DATA_PATH}" \
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
        --vllm-quantization bitsandbytes \
        --vllm-load-format bitsandbytes \
        --enforce-eager \
        --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}" \
        --attention-backend "${ATTENTION_BACKEND}" \
        --offset "${current_offset}" \
        --limit "${shard_limit}"
    ) > "${OUT_DIR}/logs/reason_8k_shard_${shard_name}.log" 2>&1 &
    PIDS+=("$!")
    echo "Started raw shard ${shard_name}: gpu=${GPU_IDS[${shard_idx}]}, offset=${current_offset}, limit=${shard_limit}, pid=${PIDS[-1]}"
  fi
  current_offset=$((current_offset + shard_limit))
done
wait_for_stage "raw 8k reasoner" "${PIDS[@]}"
PIDS=()

"${PYTHON_BIN}" codex_added/scripts/21_merge_shard_predictions.py \
  --data "${DATA_PATH}" \
  --offset "${OFFSET}" \
  --limit "${LIMIT}" \
  --predictions "${RAW_SHARDS[@]}" \
  --output "${OUT_DIR}/raw_8k_merged.jsonl"

echo "Starting LoRA finalizer at $(date -u -Iseconds)"
for shard_idx in "${!GPU_IDS[@]}"; do
  shard_name="$(printf "%02d" "${shard_idx}")"
  raw_path="${OUT_DIR}/raw/shard_${shard_name}.jsonl"
  final_path="${OUT_DIR}/finalized/shard_${shard_name}.jsonl"
  FINAL_SHARDS+=("${final_path}")
  (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS[${shard_idx}]}"
    "${PYTHON_BIN}" codex_added/scripts/16_finalize_with_qwen.py \
      --data "${DATA_PATH}" \
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
      --vllm-quantization bitsandbytes \
      --vllm-load-format bitsandbytes \
      --enforce-eager \
      --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}" \
      --attention-backend "${ATTENTION_BACKEND}" \
      --max-input-tokens "${FINAL_MAX_INPUT_TOKENS}" \
      --max-new-tokens "${FINAL_MAX_NEW_TOKENS}" \
      --temperature 0.1 \
      --top-p 0.9 \
      --retry-bad-format \
      --retry-max-new-tokens "${FINAL_RETRY_MAX_NEW_TOKENS}" \
      --score
  ) > "${OUT_DIR}/logs/finalize_lora_shard_${shard_name}.log" 2>&1 &
  PIDS+=("$!")
  echo "Started finalizer shard ${shard_name}: gpu=${GPU_IDS[${shard_idx}]}, pid=${PIDS[-1]}"
done
wait_for_stage "LoRA finalizer" "${PIDS[@]}"

"${PYTHON_BIN}" codex_added/scripts/21_merge_shard_predictions.py \
  --data "${DATA_PATH}" \
  --offset "${OFFSET}" \
  --limit "${LIMIT}" \
  --predictions "${FINAL_SHARDS[@]}" \
  --output "${OUT_DIR}/lora_finalized.jsonl"

"${PYTHON_BIN}" codex_added/scripts/39_precision_form_normalize.py \
  --data "${DATA_PATH}" \
  --raw-responses "${OUT_DIR}/raw_8k_merged.jsonl" \
  --finalized-responses "${OUT_DIR}/lora_finalized.jsonl" \
  --output "${OUT_DIR}/v4_precision_form_normalized.jsonl" \
  --audit-output "${OUT_DIR}/audit/v4_precision_form_normalized_audit.json" \
  --score

"${PYTHON_BIN}" codex_added/scripts/40_deterministic_numeric_finalizer.py \
  --data "${DATA_PATH}" \
  --responses "${OUT_DIR}/v4_precision_form_normalized.jsonl" \
  --output "${OUT_DIR}/v5_numeric_finalized.jsonl" \
  --audit-output "${OUT_DIR}/audit/v5_numeric_finalized_audit.json" \
  --score

"${PYTHON_BIN}" codex_added/scripts/44_slot_aware_finalize.py \
  --data "${DATA_PATH}" \
  --responses "${OUT_DIR}/v5_numeric_finalized.jsonl" \
  --output "${OUT_DIR}/v6_slot_aware_finalized.jsonl" \
  --audit-output "${OUT_DIR}/audit/v6_slot_aware_audit.json" \
  --score

"${PYTHON_BIN}" codex_added/scripts/53_canonicalize_derivative_mcq_options.py \
  --data "${DATA_PATH}" \
  --responses "${OUT_DIR}/v6_slot_aware_finalized.jsonl" \
  --output "${OUT_DIR}/v7_derivative_mcq_canonicalized.jsonl" \
  --audit-output "${OUT_DIR}/audit/v7_derivative_mcq_canonicalizer_audit.json" \
  --score

"${PYTHON_BIN}" codex_added/scripts/32_score_stage_progression.py \
  --data "${DATA_PATH}" \
  --stage raw "${OUT_DIR}/raw_8k_merged.jsonl" \
  --stage lora "${OUT_DIR}/lora_finalized.jsonl" \
  --stage v4 "${OUT_DIR}/v4_precision_form_normalized.jsonl" \
  --stage v5 "${OUT_DIR}/v5_numeric_finalized.jsonl" \
  --stage v6 "${OUT_DIR}/v6_slot_aware_finalized.jsonl" \
  --stage v7 "${OUT_DIR}/v7_derivative_mcq_canonicalized.jsonl" \
  --output-json "${OUT_DIR}/audit/stage_progression_holdout50.json" \
  --output-md "${OUT_DIR}/audit/stage_progression_holdout50.md" \
  --scored-dir "${OUT_DIR}/scored"

{
  echo "run_id=${RUN_ID}"
  echo "data_path=${DATA_PATH}"
  echo "out_dir=${OUT_DIR}"
  echo "offset=${OFFSET}"
  echo "limit=${LIMIT}"
  echo "gpus=${GPUS}"
  echo "max_new_tokens=${MAX_NEW_TOKENS}"
  echo "reason_max_num_seqs=${REASON_MAX_NUM_SEQS}"
  echo "reason_vllm_batch_size=${REASON_VLLM_BATCH_SIZE}"
  echo "reason_vllm_batch_size_note=0 means queue the full shard while max_num_seqs controls active concurrency"
  echo "attention_backend=${ATTENTION_BACKEND}"
  echo "vllm_rpc_base_path=${VLLM_RPC_BASE_PATH}"
  echo "raw_output=${OUT_DIR}/raw_8k_merged.jsonl"
  echo "v7_output=${OUT_DIR}/v7_derivative_mcq_canonicalized.jsonl"
  echo "stage_progression=${OUT_DIR}/audit/stage_progression_holdout50.md"
  echo "completed_at=$(date -u -Iseconds)"
} > "${OUT_DIR}/RUN_INFO.txt"

echo "completed_at=$(date -u -Iseconds)"
echo "V7 output: ${OUT_DIR}/v7_derivative_mcq_canonicalized.jsonl"
echo "Progression summary: ${OUT_DIR}/audit/stage_progression_holdout50.md"
