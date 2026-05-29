#!/usr/bin/env bash
# Added by Codex: Transformers fallback for public100 raw 8k + finalizer + repair.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CODEX_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
DATA_PATH="${DATA_PATH:-codex_added/job_data/public_diagnostic_100_offset200_40mcq_40multislot_20hard.jsonl}"
SOURCE_DATA_PATH="${SOURCE_DATA_PATH:-data/public.jsonl}"
OUT_DIR="${OUT_DIR:-codex_added/results/public100_8k_lora_verify_repair_transformers_2gpu_${RUN_ID}}"
ROW_LIMIT="${ROW_LIMIT:-100}"
ROW_OFFSET="${ROW_OFFSET:-0}"
GPUS="${GPUS:-0,1}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
RAW_QUANTIZATION="${RAW_QUANTIZATION:-none}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/.hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea}"
ADAPTER_DIR="${ADAPTER_DIR:-codex_added/models/qwen3_answer_format_lora}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-vllm/bin/python}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export TRANSFORMERS_ATTENTION_IMPLEMENTATION="${TRANSFORMERS_ATTENTION_IMPLEMENTATION:-sdpa}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export TMPDIR="${TMPDIR:-${REPO_ROOT}/.tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${REPO_ROOT}/.pip-cache}"

validate_gpu_csv() {
  local gpus_csv="$1"
  local gpu_ids=()
  IFS=',' read -r -a gpu_ids <<< "${gpus_csv}"
  if [[ "${#gpu_ids[@]}" -lt 1 ]]; then
    echo "GPUS must contain at least one GPU id." >&2
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

validate_gpu_csv "${GPUS}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/raw" "${OUT_DIR}/finalized" "${OUT_DIR}/verify_repair" "${TMPDIR}"
exec > >(tee -a "${OUT_DIR}/public100_transformers_full_pipeline_run.log") 2>&1

if [[ ! -f "${DATA_PATH}" ]]; then
  "${PYTHON_BIN}" codex_added/scripts/30_build_public_diagnostic_slice.py \
    --data "${SOURCE_DATA_PATH}" \
    --output "${DATA_PATH}"
fi

echo "run_id=${RUN_ID}"
echo "data_path=${DATA_PATH}"
echo "row_offset=${ROW_OFFSET}"
echo "row_limit=${ROW_LIMIT}"
echo "gpus=${GPUS}"
echo "out_dir=${OUT_DIR}"
echo "backend=transformers"
echo "raw_quantization=${RAW_QUANTIZATION}"
echo "max_gpu_used_mib=${MAX_GPU_USED_MIB}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "started_at=$(date -u -Iseconds)"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing model path: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "Missing finalizer adapter: ${ADAPTER_DIR}" >&2
  exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
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
fi

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
SHARD_COUNT="${#GPU_IDS[@]}"
BASE_ROWS=$((ROW_LIMIT / SHARD_COUNT))
EXTRA_ROWS=$((ROW_LIMIT % SHARD_COUNT))

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
    echo "${stage} failed; check ${OUT_DIR}/logs" >&2
    exit 1
  fi
}

declare -a RAW_SHARDS FINAL_SHARDS REPAIR_SHARDS PIDS

current_offset="${ROW_OFFSET}"
echo "Starting raw 8k Transformers shards at $(date -u -Iseconds)"
for shard_idx in "${!GPU_IDS[@]}"; do
  shard_limit="${BASE_ROWS}"
  if [[ "${shard_idx}" -lt "${EXTRA_ROWS}" ]]; then
    shard_limit=$((shard_limit + 1))
  fi
  shard_name="$(printf "%02d" "${shard_idx}")"
  raw_path="${OUT_DIR}/raw/shard_${shard_name}.jsonl"
  RAW_SHARDS+=("${raw_path}")
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
      --backend transformers \
      --model-id "${MODEL_PATH}" \
      --quantization "${RAW_QUANTIZATION}" \
      --batch-size 1 \
      --offset "${current_offset}" \
      --limit "${shard_limit}"
  ) > "${OUT_DIR}/logs/reason_8k_transformers_shard_${shard_name}.log" 2>&1 &
  PIDS+=("$!")
  echo "Started raw shard ${shard_name}: gpu=${GPU_IDS[${shard_idx}]}, offset=${current_offset}, limit=${shard_limit}, pid=${PIDS[-1]}"
  current_offset=$((current_offset + shard_limit))
done
wait_for_stage "raw 8k Transformers" "${PIDS[@]}"
PIDS=()

"${PYTHON_BIN}" codex_added/scripts/21_merge_shard_predictions.py \
  --data "${DATA_PATH}" \
  --offset "${ROW_OFFSET}" \
  --limit "${ROW_LIMIT}" \
  --predictions "${RAW_SHARDS[@]}" \
  --output "${OUT_DIR}/raw_8k_merged.jsonl"

echo "Starting LoRA finalizer Transformers shards at $(date -u -Iseconds)"
for shard_idx in "${!GPU_IDS[@]}"; do
  shard_name="$(printf "%02d" "${shard_idx}")"
  final_path="${OUT_DIR}/finalized/shard_${shard_name}.jsonl"
  FINAL_SHARDS+=("${final_path}")
  (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS[${shard_idx}]}"
    "${PYTHON_BIN}" codex_added/scripts/16_finalize_with_qwen.py \
      --data "${DATA_PATH}" \
      --responses "${OUT_DIR}/raw/shard_${shard_name}.jsonl" \
      --output "${final_path}" \
      --model-id "${MODEL_PATH}" \
      --adapter-dir "${ADAPTER_DIR}" \
      --backend transformers \
      --max-input-tokens "${FINAL_MAX_INPUT_TOKENS:-4096}" \
      --max-new-tokens "${FINAL_MAX_NEW_TOKENS:-64}" \
      --temperature 0.1 \
      --top-p 0.9 \
      --retry-bad-format \
      --retry-max-new-tokens "${FINAL_RETRY_MAX_NEW_TOKENS:-128}" \
      --score
  ) > "${OUT_DIR}/logs/finalize_lora_transformers_shard_${shard_name}.log" 2>&1 &
  PIDS+=("$!")
  echo "Started finalizer shard ${shard_name}: gpu=${GPU_IDS[${shard_idx}]}, pid=${PIDS[-1]}"
done
wait_for_stage "LoRA finalizer Transformers" "${PIDS[@]}"
PIDS=()

"${PYTHON_BIN}" codex_added/scripts/21_merge_shard_predictions.py \
  --data "${DATA_PATH}" \
  --offset "${ROW_OFFSET}" \
  --limit "${ROW_LIMIT}" \
  --predictions "${FINAL_SHARDS[@]}" \
  --output "${OUT_DIR}/finalized_8k_merged.jsonl"

echo "Starting verifier/repair Transformers shards at $(date -u -Iseconds)"
for shard_idx in "${!GPU_IDS[@]}"; do
  shard_name="$(printf "%02d" "${shard_idx}")"
  repair_path="${OUT_DIR}/verify_repair/shard_${shard_name}.jsonl"
  REPAIR_SHARDS+=("${repair_path}")
  (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS[${shard_idx}]}"
    "${PYTHON_BIN}" codex_added/scripts/31_verify_repair_with_qwen.py \
      --data "${DATA_PATH}" \
      --raw-responses "${OUT_DIR}/raw/shard_${shard_name}.jsonl" \
      --finalized-responses "${OUT_DIR}/finalized/shard_${shard_name}.jsonl" \
      --output "${repair_path}" \
      --model-id "${MODEL_PATH}" \
      --backend transformers \
      --max-input-tokens "${VERIFY_MAX_INPUT_TOKENS:-6144}" \
      --max-new-tokens "${VERIFY_MAX_NEW_TOKENS:-1024}" \
      --temperature 0.1 \
      --top-p 0.9 \
      --score
  ) > "${OUT_DIR}/logs/verify_repair_transformers_shard_${shard_name}.log" 2>&1 &
  PIDS+=("$!")
  echo "Started verifier/repair shard ${shard_name}: gpu=${GPU_IDS[${shard_idx}]}, pid=${PIDS[-1]}"
done
wait_for_stage "verifier/repair Transformers" "${PIDS[@]}"

"${PYTHON_BIN}" codex_added/scripts/21_merge_shard_predictions.py \
  --data "${DATA_PATH}" \
  --offset "${ROW_OFFSET}" \
  --limit "${ROW_LIMIT}" \
  --predictions "${REPAIR_SHARDS[@]}" \
  --output "${OUT_DIR}/verify_repair_merged.jsonl"

"${PYTHON_BIN}" codex_added/scripts/32_score_stage_progression.py \
  --data "${DATA_PATH}" \
  --stage raw_8k "${OUT_DIR}/raw_8k_merged.jsonl" \
  --stage lora_finalizer "${OUT_DIR}/finalized_8k_merged.jsonl" \
  --stage verify_repair "${OUT_DIR}/verify_repair_merged.jsonl" \
  --output-json "${OUT_DIR}/stage_progression.json" \
  --output-md "${OUT_DIR}/stage_progression.md" \
  --scored-dir "${OUT_DIR}/scored"

{
  echo "run_id=${RUN_ID}"
  echo "data_path=${DATA_PATH}"
  echo "raw_output=${OUT_DIR}/raw_8k_merged.jsonl"
  echo "finalized_output=${OUT_DIR}/finalized_8k_merged.jsonl"
  echo "verify_repair_output=${OUT_DIR}/verify_repair_merged.jsonl"
  echo "stage_progression=${OUT_DIR}/stage_progression.md"
  echo "completed_at=$(date -u -Iseconds)"
} > "${OUT_DIR}/RUN_INFO.txt"

echo "Done at $(date -u -Iseconds)"
echo "Progression summary: ${OUT_DIR}/stage_progression.md"
