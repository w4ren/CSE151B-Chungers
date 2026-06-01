#!/usr/bin/env bash
# Run conflict-only reranking and finalization without the final selection policy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_PATH="${1:-data/private.jsonl}"
OUT_DIR="${2:-codex_added/results/third_best_private}"
SUBMISSION_PATH="${3:-codex_added/submissions/third_best_submission.csv}"
ADAPTER_DIR="${ADAPTER_DIR:-codex_added/models/qwen3_answer_format_lora}"
OFFSET="${OFFSET:-}"
LIMIT="${LIMIT:-}"
SCORE="${SCORE:-0}"

cd "${REPO_ROOT}"

mkdir -p "${OUT_DIR}"
mkdir -p "$(dirname "${SUBMISSION_PATH}")"

OFFSET_ARGS=()
if [[ -n "${OFFSET}" ]]; then
  OFFSET_ARGS=(--offset "${OFFSET}")
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

SCORE_ARGS=()
if [[ "${SCORE}" == "1" ]]; then
  SCORE_ARGS=(--score)
fi

python codex_added/scripts/10_run_prompt_sweep.py \
  --input "${DATA_PATH}" \
  --output "${OUT_DIR}/sweep_2048.jsonl" \
  "${OFFSET_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  --variants final_box_only \
  --num-samples 1 \
  --do-sample \
  --quantization none \
  --max-new-tokens 2048 \
  --batch-size 1 \
  --answer-key-mode strict

python codex_added/scripts/16_finalize_with_qwen.py \
  --data "${DATA_PATH}" \
  --responses "${OUT_DIR}/sweep_2048.jsonl" \
  --output "${OUT_DIR}/finalized_2048.jsonl" \
  --adapter-dir "${ADAPTER_DIR}" \
  --max-new-tokens 64 \
  --temperature 0.1 \
  "${SCORE_ARGS[@]}"

python codex_added/scripts/10_run_prompt_sweep.py \
  --input "${DATA_PATH}" \
  --output "${OUT_DIR}/sweep_1024.jsonl" \
  "${OFFSET_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  --variants final_box_only \
  --num-samples 1 \
  --do-sample \
  --quantization none \
  --max-new-tokens 1024 \
  --batch-size 1 \
  --answer-key-mode strict

python codex_added/scripts/16_finalize_with_qwen.py \
  --data "${DATA_PATH}" \
  --responses "${OUT_DIR}/sweep_1024.jsonl" \
  --output "${OUT_DIR}/finalized_1024.jsonl" \
  --adapter-dir "${ADAPTER_DIR}" \
  --max-new-tokens 64 \
  --temperature 0.1 \
  "${SCORE_ARGS[@]}"

python codex_added/scripts/17_rerank_with_qwen.py \
  --data "${DATA_PATH}" \
  --responses "${OUT_DIR}/finalized_1024.jsonl" "${OUT_DIR}/finalized_2048.jsonl" \
  --output "${OUT_DIR}/reranked_conflicts.jsonl" \
  "${OFFSET_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  --max-candidates 8 \
  --dedupe-answer-keys \
  --only-conflicts \
  --assistant-mode think \
  --max-new-tokens 1536 \
  --temperature 0.1 \
  "${SCORE_ARGS[@]}"

python codex_added/scripts/16_finalize_with_qwen.py \
  --data "${DATA_PATH}" \
  --responses "${OUT_DIR}/reranked_conflicts.jsonl" \
  --output "${OUT_DIR}/selected.jsonl" \
  --adapter-dir "${ADAPTER_DIR}" \
  --max-new-tokens 64 \
  --temperature 0.1 \
  "${SCORE_ARGS[@]}"

if [[ -z "${OFFSET}" && -z "${LIMIT}" ]]; then
  python codex_added/scripts/make_submission.py \
    --data "${DATA_PATH}" \
    --predictions "${OUT_DIR}/selected.jsonl" \
    --output "${SUBMISSION_PATH}"
  echo "Wrote submission CSV to ${SUBMISSION_PATH}"
else
  echo "OFFSET or LIMIT is set, so submission CSV was skipped because it would not contain every id."
fi
