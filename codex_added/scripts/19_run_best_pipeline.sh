#!/usr/bin/env bash
# Added by Codex: reproducible best-known Qwen-only pipeline; not part of the original starter repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_PATH="${1:-data/private.jsonl}"
OUT_DIR="${2:-codex_added/results/best_private}"
SUBMISSION_PATH="${3:-codex_added/submissions/best_submission.csv}"
ADAPTER_DIR="${ADAPTER_DIR:-codex_added/models/qwen3_answer_format_lora}"
BACKEND="${BACKEND:-vllm}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv-vllm/bin/python" ]]; then
    PYTHON=".venv-vllm/bin/python"
  else
    PYTHON="python"
  fi
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${REPO_ROOT}/.vllm-cache}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing data file: ${DATA_PATH}" >&2
  echo "Usage: $0 data/private.jsonl codex_added/results/best_private codex_added/submissions/best_submission.csv" >&2
  exit 1
fi

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "Missing LoRA adapter directory: ${ADAPTER_DIR}" >&2
  echo "Train it with codex_added/scripts/13_train_lora_sft.py before running this pipeline." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
mkdir -p "$(dirname "${SUBMISSION_PATH}")"

OFFSET_ARGS=()
if [[ -n "${OFFSET:-}" ]]; then
  OFFSET_ARGS=(--offset "${OFFSET}")
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

PARTIAL_RUN=0
if [[ -n "${OFFSET:-}" && "${OFFSET}" != "0" ]]; then
  PARTIAL_RUN=1
fi
if [[ -n "${LIMIT:-}" ]]; then
  PARTIAL_RUN=1
fi

SCORE_ARGS=()
if [[ "${SCORE:-0}" == "1" ]]; then
  SCORE_ARGS=(--score)
fi

MODEL_ARGS=()
if [[ -n "${MODEL_ID:-}" ]]; then
  MODEL_ARGS=(--model-id "${MODEL_ID}")
fi

BACKEND_ARGS=(--backend "${BACKEND}")
SWEEP_BATCH_ARGS=()
if [[ "${BACKEND}" == "vllm" ]]; then
  BACKEND_ARGS+=(
    --dtype "${VLLM_DTYPE:-float16}"
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
    --max-model-len "${VLLM_MAX_MODEL_LEN:-6144}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-16}"
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
  )
  if [[ -n "${VLLM_BATCH_SIZE:-}" ]]; then
    BACKEND_ARGS+=(--vllm-batch-size "${VLLM_BATCH_SIZE}")
  fi
  if [[ -n "${VLLM_QUANTIZATION:-}" ]]; then
    BACKEND_ARGS+=(--vllm-quantization "${VLLM_QUANTIZATION}" --vllm-load-format "${VLLM_LOAD_FORMAT:-auto}")
  fi
  if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
    BACKEND_ARGS+=(--enforce-eager)
  fi
  if [[ -n "${SAFETENSORS_LOAD_STRATEGY:-}" ]]; then
    BACKEND_ARGS+=(--safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}")
  fi
  if [[ "${VLLM_DISABLE_PREFIX_CACHING:-0}" == "1" ]]; then
    BACKEND_ARGS+=(--disable-prefix-caching)
  fi

  SUBMISSION_ARGS=()
  if [[ "${PARTIAL_RUN}" != "0" ]]; then
    SUBMISSION_ARGS=(--no-write-submission)
  fi

  "${PYTHON}" codex_added/scripts/23_run_best_pipeline_vllm_single.py \
    --data "${DATA_PATH}" \
    --out-dir "${OUT_DIR}" \
    --submission "${SUBMISSION_PATH}" \
    --adapter-dir "${ADAPTER_DIR}" \
    "${OFFSET_ARGS[@]}" \
    "${LIMIT_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${BACKEND_ARGS[@]}" \
    "${SUBMISSION_ARGS[@]}" \
    --retry-bad-format \
    --retry-max-new-tokens "${FINALIZER_RETRY_MAX_NEW_TOKENS:-128}" \
    "${SCORE_ARGS[@]}"
  exit 0
else
  SWEEP_BATCH_ARGS=(--batch-size "${TRANSFORMERS_BATCH_SIZE:-1}")
fi

"${PYTHON}" codex_added/scripts/10_run_prompt_sweep.py \
  --input "${DATA_PATH}" \
  --output "${OUT_DIR}/sweep_2048.jsonl" \
  "${OFFSET_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  "${BACKEND_ARGS[@]}" \
  "${SWEEP_BATCH_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --variants final_box_only \
  --num-samples 1 \
  --do-sample \
  --quantization none \
  --max-new-tokens 2048 \
  --answer-key-mode strict

"${PYTHON}" codex_added/scripts/16_finalize_with_qwen.py \
  --data "${DATA_PATH}" \
  --responses "${OUT_DIR}/sweep_2048.jsonl" \
  --output "${OUT_DIR}/finalized_2048.jsonl" \
  "${BACKEND_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --max-input-tokens "${FINALIZER_MAX_INPUT_TOKENS:-4096}" \
  --max-new-tokens 64 \
  --temperature 0.1 \
  --retry-bad-format \
  --retry-max-new-tokens "${FINALIZER_RETRY_MAX_NEW_TOKENS:-128}" \
  "${SCORE_ARGS[@]}"

"${PYTHON}" codex_added/scripts/10_run_prompt_sweep.py \
  --input "${DATA_PATH}" \
  --output "${OUT_DIR}/sweep_1024.jsonl" \
  "${OFFSET_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  "${BACKEND_ARGS[@]}" \
  "${SWEEP_BATCH_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --variants final_box_only \
  --num-samples 1 \
  --do-sample \
  --quantization none \
  --max-new-tokens 1024 \
  --answer-key-mode strict

"${PYTHON}" codex_added/scripts/16_finalize_with_qwen.py \
  --data "${DATA_PATH}" \
  --responses "${OUT_DIR}/sweep_1024.jsonl" \
  --output "${OUT_DIR}/finalized_1024.jsonl" \
  "${BACKEND_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --max-input-tokens "${FINALIZER_MAX_INPUT_TOKENS:-4096}" \
  --max-new-tokens 64 \
  --temperature 0.1 \
  --retry-bad-format \
  --retry-max-new-tokens "${FINALIZER_RETRY_MAX_NEW_TOKENS:-128}" \
  "${SCORE_ARGS[@]}"

"${PYTHON}" codex_added/scripts/17_rerank_with_qwen.py \
  --data "${DATA_PATH}" \
  --responses "${OUT_DIR}/finalized_1024.jsonl" "${OUT_DIR}/finalized_2048.jsonl" \
  --output "${OUT_DIR}/reranked_conflicts.jsonl" \
  "${OFFSET_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  "${BACKEND_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --max-candidates 8 \
  --dedupe-answer-keys \
  --only-conflicts \
  --assistant-mode "${RERANK_ASSISTANT_MODE:-think}" \
  --max-input-tokens "${RERANK_MAX_INPUT_TOKENS:-4096}" \
  --max-new-tokens "${RERANK_MAX_NEW_TOKENS:-1536}" \
  --temperature 0.1 \
  "${SCORE_ARGS[@]}"

"${PYTHON}" codex_added/scripts/16_finalize_with_qwen.py \
  --data "${DATA_PATH}" \
  --responses "${OUT_DIR}/reranked_conflicts.jsonl" \
  --output "${OUT_DIR}/finalized_reranked_conflicts.jsonl" \
  "${BACKEND_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --max-input-tokens "${FINALIZER_MAX_INPUT_TOKENS:-4096}" \
  --max-new-tokens 64 \
  --temperature 0.1 \
  --retry-bad-format \
  --retry-max-new-tokens "${FINALIZER_RETRY_MAX_NEW_TOKENS:-128}" \
  "${SCORE_ARGS[@]}"

"${PYTHON}" codex_added/scripts/18_select_predictions.py \
  --data "${DATA_PATH}" \
  --base "${OUT_DIR}/finalized_2048.jsonl" \
  --override "${OUT_DIR}/finalized_reranked_conflicts.jsonl" \
  --output "${OUT_DIR}/selected.jsonl" \
  "${OFFSET_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  --policy free_form_override \
  "${SCORE_ARGS[@]}"

echo "Wrote selected JSONL to ${OUT_DIR}/selected.jsonl"
if [[ "${PARTIAL_RUN}" == "0" ]]; then
  "${PYTHON}" codex_added/scripts/make_submission.py \
    --data "${DATA_PATH}" \
    --predictions "${OUT_DIR}/selected.jsonl" \
    --output "${SUBMISSION_PATH}"
  echo "Wrote submission CSV to ${SUBMISSION_PATH}"
else
  echo "OFFSET or LIMIT is set, so submission CSV was skipped because it would not contain every id."
fi
