#!/usr/bin/env bash
# Added by Codex: run the best current 8k vLLM + LoRA finalizer pipeline on the fixed eval-40 slice.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CODEX_ROOT}/.." && pwd)"
ARCHIVE2_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_PATH="${1:-codex_added/job_data/public_stratified_40.jsonl}"
OUT_DIR="${2:-codex_added/results/vllm_eval40_8k_finalized_2gpu}"
LIMIT="${LIMIT:-40}"
OFFSET="${OFFSET:-0}"
GPUS="${GPUS:-0,1}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
REASON_MAX_MODEL_LEN="${REASON_MAX_MODEL_LEN:-10240}"
REASON_GPU_UTIL="${REASON_GPU_UTIL:-0.95}"
REASON_MAX_NUM_SEQS="${REASON_MAX_NUM_SEQS:-2}"
REASON_MAX_NUM_BATCHED_TOKENS="${REASON_MAX_NUM_BATCHED_TOKENS:-10240}"
REASON_VLLM_BATCH_SIZE="${REASON_VLLM_BATCH_SIZE:-0}"

FINAL_MAX_MODEL_LEN="${FINAL_MAX_MODEL_LEN:-6144}"
FINAL_GPU_UTIL="${FINAL_GPU_UTIL:-0.90}"
FINAL_MAX_NUM_SEQS="${FINAL_MAX_NUM_SEQS:-4}"
FINAL_MAX_NUM_BATCHED_TOKENS="${FINAL_MAX_NUM_BATCHED_TOKENS:-6144}"
FINAL_VLLM_BATCH_SIZE="${FINAL_VLLM_BATCH_SIZE:-8}"
FINAL_MAX_INPUT_TOKENS="${FINAL_MAX_INPUT_TOKENS:-4096}"
FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS:-64}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/.hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea}"
ADAPTER_DIR="${ADAPTER_DIR:-codex_added/models/qwen3_answer_format_lora}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-vllm/bin/python}"

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
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/raw" "${OUT_DIR}/finalized" \
    "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${VLLM_RPC_BASE_PATH}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python environment: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing local model snapshot: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${DATA_PATH}" ]]; then
    echo "Missing data slice: ${DATA_PATH}" >&2
    exit 1
fi
if [[ ! -d "${ADAPTER_DIR}" ]]; then
    echo "Missing finalizer adapter: ${ADAPTER_DIR}" >&2
    exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
    echo "No GPUs configured in GPUS=${GPUS}" >&2
    exit 1
fi

SHARD_COUNT="${#GPU_IDS[@]}"
BASE_ROWS=$((LIMIT / SHARD_COUNT))
EXTRA_ROWS=$((LIMIT % SHARD_COUNT))

declare -a RAW_SHARDS
declare -a FINAL_SHARDS
declare -a PIDS

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

echo "Data: ${DATA_PATH}"
echo "Output: ${OUT_DIR}"
echo "GPUs: ${GPUS}"
echo "Limit/offset: ${LIMIT}/${OFFSET}"
echo "Reasoner: ${MAX_NEW_TOKENS} tokens, max_model_len=${REASON_MAX_MODEL_LEN}, max_num_seqs=${REASON_MAX_NUM_SEQS}, bnb vLLM"
echo "Finalizer: LoRA, max_input_tokens=${FINAL_MAX_INPUT_TOKENS}, max_new_tokens=${FINAL_MAX_NEW_TOKENS}, bnb vLLM"

current_offset="${OFFSET}"
for shard_idx in "${!GPU_IDS[@]}"; do
    shard_limit="${BASE_ROWS}"
    if [[ "${shard_idx}" -lt "${EXTRA_ROWS}" ]]; then
        shard_limit=$((shard_limit + 1))
    fi
    shard_name="$(printf "%02d" "${shard_idx}")"
    raw_path="${OUT_DIR}/raw/shard_${shard_name}.jsonl"
    log_path="${OUT_DIR}/logs/reason_8k_shard_${shard_name}.log"
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
                --offset "${current_offset}" \
                --limit "${shard_limit}"
        ) > "${log_path}" 2>&1 &
        PIDS+=("$!")
        echo "Started reason shard ${shard_name} on GPU ${GPU_IDS[${shard_idx}]}: offset=${current_offset}, limit=${shard_limit}, pid=${PIDS[-1]}"
    fi
    current_offset=$((current_offset + shard_limit))
done

wait_for_stage "8k reasoner" "${PIDS[@]}"
PIDS=()

"${PYTHON_BIN}" codex_added/scripts/21_merge_shard_predictions.py \
    --data "${DATA_PATH}" \
    --offset "${OFFSET}" \
    --limit "${LIMIT}" \
    --predictions "${RAW_SHARDS[@]}" \
    --output "${OUT_DIR}/raw_8k_merged.jsonl"

for shard_idx in "${!GPU_IDS[@]}"; do
    shard_name="$(printf "%02d" "${shard_idx}")"
    raw_path="${OUT_DIR}/raw/shard_${shard_name}.jsonl"
    final_path="${OUT_DIR}/finalized/shard_${shard_name}.jsonl"
    log_path="${OUT_DIR}/logs/finalize_lora_shard_${shard_name}.log"
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
            --max-input-tokens "${FINAL_MAX_INPUT_TOKENS}" \
            --max-new-tokens "${FINAL_MAX_NEW_TOKENS}" \
            --temperature 0.1 \
            --top-p 0.9 \
            --retry-bad-format \
            --retry-max-new-tokens "${FINAL_RETRY_MAX_NEW_TOKENS:-128}" \
            --score
    ) > "${log_path}" 2>&1 &
    PIDS+=("$!")
    echo "Started finalizer shard ${shard_name} on GPU ${GPU_IDS[${shard_idx}]}: pid=${PIDS[-1]}"
done

wait_for_stage "LoRA finalizer" "${PIDS[@]}"

"${PYTHON_BIN}" codex_added/scripts/21_merge_shard_predictions.py \
    --data "${DATA_PATH}" \
    --offset "${OFFSET}" \
    --limit "${LIMIT}" \
    --predictions "${FINAL_SHARDS[@]}" \
    --output "${OUT_DIR}/finalized_8k_merged.jsonl"

"${PYTHON_BIN}" codex_added/scripts/26_select_budget_candidates.py \
    --data "${DATA_PATH}" \
    --offset "${OFFSET}" \
    --limit "${LIMIT}" \
    --responses "${OUT_DIR}/finalized_8k_merged.jsonl" \
    --labels 8k_lora_finalized \
    --output "${OUT_DIR}/selected_8k_lora_finalized.jsonl" \
    --score

echo "Done. Main scored output: ${OUT_DIR}/selected_8k_lora_finalized.jsonl"
