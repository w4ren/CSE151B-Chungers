#!/usr/bin/env bash
# Added by Codex: MCQ-only best-of-N experiment for the public200 audit slice.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CODEX_ROOT}/.." && pwd)"
ARCHIVE2_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_PATH="${DATA_PATH:-codex_added/job_data/public200_first100_plus_diagnostic100.jsonl}"
BASELINE_PATH="${BASELINE_PATH:-codex_added/results/public200_strict_audit_2gpu_20260528_092015/v5_numeric_finalized_merged.jsonl}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-codex_added/results/public200_mcq_bestofN_${RUN_STAMP}}"
MCQ_DATA="${OUT_DIR}/public200_mcq.jsonl"

NUM_SAMPLES="${NUM_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
REASON_MAX_MODEL_LEN="${REASON_MAX_MODEL_LEN:-6144}"
REASON_GPU_UTIL="${REASON_GPU_UTIL:-0.90}"
REASON_MAX_NUM_SEQS="${REASON_MAX_NUM_SEQS:-1}"
REASON_MAX_NUM_BATCHED_TOKENS="${REASON_MAX_NUM_BATCHED_TOKENS:-6144}"
REASON_VLLM_BATCH_SIZE="${REASON_VLLM_BATCH_SIZE:-1}"

FINAL_MAX_MODEL_LEN="${FINAL_MAX_MODEL_LEN:-4096}"
FINAL_GPU_UTIL="${FINAL_GPU_UTIL:-0.88}"
FINAL_MAX_NUM_SEQS="${FINAL_MAX_NUM_SEQS:-4}"
FINAL_MAX_NUM_BATCHED_TOKENS="${FINAL_MAX_NUM_BATCHED_TOKENS:-4096}"
FINAL_VLLM_BATCH_SIZE="${FINAL_VLLM_BATCH_SIZE:-4}"
FINAL_MAX_INPUT_TOKENS="${FINAL_MAX_INPUT_TOKENS:-3072}"
FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS:-64}"

GPU="${GPU:-0}"
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
export CUDA_VISIBLE_DEVICES="${GPU}"

mkdir -p "${OUT_DIR}/logs" "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${VLLM_RPC_BASE_PATH}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python environment: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing local model snapshot: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${DATA_PATH}" ]]; then
    echo "Missing data: ${DATA_PATH}" >&2
    exit 1
fi
if [[ ! -f "${BASELINE_PATH}" ]]; then
    echo "Missing baseline predictions: ${BASELINE_PATH}" >&2
    exit 1
fi
if [[ ! -d "${ADAPTER_DIR}" ]]; then
    echo "Missing finalizer adapter: ${ADAPTER_DIR}" >&2
    exit 1
fi

echo "Data: ${DATA_PATH}"
echo "Output: ${OUT_DIR}"
echo "Samples per MCQ: ${NUM_SAMPLES}"
echo "GPU: ${GPU}"

"${PYTHON_BIN}" codex_added/scripts/41_build_mcq_slice.py \
    --data "${DATA_PATH}" \
    --output "${MCQ_DATA}"

"${PYTHON_BIN}" codex_added/scripts/10_run_prompt_sweep.py \
    --input "${MCQ_DATA}" \
    --output "${OUT_DIR}/raw_mcq_bestofN.jsonl" \
    --variants final_box_only \
    --num-samples "${NUM_SAMPLES}" \
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
    > "${OUT_DIR}/logs/reason_mcq_bestofN.log" 2>&1

"${PYTHON_BIN}" codex_added/scripts/16_finalize_with_qwen.py \
    --data "${MCQ_DATA}" \
    --responses "${OUT_DIR}/raw_mcq_bestofN.jsonl" \
    --output "${OUT_DIR}/finalized_mcq_bestofN.jsonl" \
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
    --score \
    > "${OUT_DIR}/logs/finalize_mcq_bestofN.log" 2>&1

"${PYTHON_BIN}" codex_added/scripts/25_analyze_candidate_pool.py \
    --data "${MCQ_DATA}" \
    --responses "${OUT_DIR}/finalized_mcq_bestofN.jsonl" \
    --labels finalized_bestofN \
    --output "${OUT_DIR}/candidate_diagnostics.jsonl" \
    --summary-output "${OUT_DIR}/candidate_summary.json"

"${PYTHON_BIN}" codex_added/scripts/42_select_mcq_bestof.py \
    --data "${MCQ_DATA}" \
    --responses "${OUT_DIR}/finalized_mcq_bestofN.jsonl" \
    --baseline "${BASELINE_PATH}" \
    --output "${OUT_DIR}/selected_mcq_consensus_or_baseline.jsonl" \
    --details-output "${OUT_DIR}/selection_details.jsonl" \
    --summary-output "${OUT_DIR}/selection_summary.json" \
    --policy consensus_or_baseline \
    --score

echo "Done. Summary: ${OUT_DIR}/selection_summary.json"
