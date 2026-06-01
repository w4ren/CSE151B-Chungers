#!/usr/bin/env bash
# Added by Codex: MCQ-only short/direct best-of-N experiment and conservative V6 merge.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CODEX_ROOT}/.." && pwd)"
ARCHIVE2_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_PATH="${DATA_PATH:-codex_added/job_data/public200_first100_plus_diagnostic100.jsonl}"
BASELINE_PATH="${BASELINE_PATH:-codex_added/results/public200_strict_audit_2gpu_20260528_092015/v6_slot_aware_finalized_merged.jsonl}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-codex_added/results/public200_mcq_direct_bestofN_${RUN_STAMP}}"
MCQ_DATA="${OUT_DIR}/public200_mcq.jsonl"
MCQ_LIMIT="${MCQ_LIMIT:-}"

NUM_SAMPLES="${NUM_SAMPLES:-4}"
MIN_CONSENSUS="${MIN_CONSENSUS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
GPU="${GPU:-0}"

VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.88}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-3072}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-2}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/.hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea}"
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

echo "Data: ${DATA_PATH}"
echo "Baseline: ${BASELINE_PATH}"
echo "Output: ${OUT_DIR}"
echo "MCQ short samples: ${NUM_SAMPLES}, min consensus: ${MIN_CONSENSUS}"
echo "GPU: ${GPU}"

BUILD_MCQS=(
    "${PYTHON_BIN}" codex_added/scripts/41_build_mcq_slice.py
    --data "${DATA_PATH}"
    --output "${MCQ_DATA}"
)
if [[ -n "${MCQ_LIMIT}" ]]; then
    BUILD_MCQS+=(--limit "${MCQ_LIMIT}")
fi
"${BUILD_MCQS[@]}"

"${PYTHON_BIN}" codex_added/scripts/10_run_prompt_sweep.py \
    --input "${MCQ_DATA}" \
    --output "${OUT_DIR}/raw_mcq_direct_bestofN.jsonl" \
    --variants mcq_direct_vote \
    --num-samples "${NUM_SAMPLES}" \
    --do-sample \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.7 \
    --top-p 0.9 \
    --top-k 20 \
    --assistant-mode direct \
    --answer-key-mode strict \
    --backend vllm \
    --model-id "${MODEL_PATH}" \
    --quantization none \
    --dtype float16 \
    --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    --vllm-batch-size "${VLLM_BATCH_SIZE}" \
    --vllm-quantization bitsandbytes \
    --vllm-load-format bitsandbytes \
    --enforce-eager \
    --safetensors-load-strategy prefetch \
    > "${OUT_DIR}/logs/reason_mcq_direct_bestofN.log" 2>&1

"${PYTHON_BIN}" codex_added/scripts/25_analyze_candidate_pool.py \
    --data "${MCQ_DATA}" \
    --responses "${OUT_DIR}/raw_mcq_direct_bestofN.jsonl" \
    --labels mcq_direct_bestofN \
    --output "${OUT_DIR}/candidate_diagnostics.jsonl" \
    --summary-output "${OUT_DIR}/candidate_summary.json"

"${PYTHON_BIN}" codex_added/scripts/42_select_mcq_bestof.py \
    --data "${MCQ_DATA}" \
    --responses "${OUT_DIR}/raw_mcq_direct_bestofN.jsonl" \
    --baseline "${BASELINE_PATH}" \
    --output "${OUT_DIR}/selected_mcq_consensus_or_baseline.jsonl" \
    --details-output "${OUT_DIR}/selection_details.jsonl" \
    --summary-output "${OUT_DIR}/selection_summary.json" \
    --policy consensus_or_baseline \
    --min-consensus "${MIN_CONSENSUS}" \
    --score

"${PYTHON_BIN}" codex_added/scripts/18_select_predictions.py \
    --data "${DATA_PATH}" \
    --base "${BASELINE_PATH}" \
    --override "${OUT_DIR}/selected_mcq_consensus_or_baseline.jsonl" \
    --output "${OUT_DIR}/hybrid_v6_mcq_direct_consensus.jsonl" \
    --policy mcq_override \
    --score

"${PYTHON_BIN}" codex_added/scripts/46_score_mcq_direct_hybrid.py \
    --run-dir "${OUT_DIR}" \
    > "${OUT_DIR}/hybrid_summary.json"

echo "Done."
echo "Selection summary: ${OUT_DIR}/selection_summary.json"
echo "Hybrid summary: ${OUT_DIR}/hybrid_summary.json"
echo "Hybrid output: ${OUT_DIR}/hybrid_v6_mcq_direct_consensus.jsonl"
