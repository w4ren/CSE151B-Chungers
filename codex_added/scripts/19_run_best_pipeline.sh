#!/usr/bin/env bash
# Added by Codex: reproducible best-known Qwen-only pipeline; not part of the original starter repository.

set -euo pipefail

DATA_PATH="${1:-data/private.jsonl}"
OUT_DIR="${2:-codex_added/results/best_private}"
SUBMISSION_PATH="${3:-codex_added/submissions/best_submission.csv}"
ADAPTER_DIR="${ADAPTER_DIR:-codex_added/models/qwen3_answer_format_lora}"

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

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

SCORE_ARGS=()
if [[ "${SCORE:-0}" == "1" ]]; then
  SCORE_ARGS=(--score)
fi

python codex_added/scripts/10_run_prompt_sweep.py \
  --input "${DATA_PATH}" \
  --output "${OUT_DIR}/sweep_2048.jsonl" \
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
  --output "${OUT_DIR}/finalized_reranked_conflicts.jsonl" \
  --adapter-dir "${ADAPTER_DIR}" \
  --max-new-tokens 64 \
  --temperature 0.1 \
  "${SCORE_ARGS[@]}"

python codex_added/scripts/18_select_predictions.py \
  --data "${DATA_PATH}" \
  --base "${OUT_DIR}/finalized_2048.jsonl" \
  --override "${OUT_DIR}/finalized_reranked_conflicts.jsonl" \
  --output "${OUT_DIR}/selected.jsonl" \
  "${LIMIT_ARGS[@]}" \
  --policy free_form_override \
  "${SCORE_ARGS[@]}"

echo "Wrote selected JSONL to ${OUT_DIR}/selected.jsonl"
if [[ -z "${LIMIT:-}" ]]; then
  python codex_added/scripts/make_submission.py \
    --data "${DATA_PATH}" \
    --predictions "${OUT_DIR}/selected.jsonl" \
    --output "${SUBMISSION_PATH}"
  echo "Wrote submission CSV to ${SUBMISSION_PATH}"
else
  echo "LIMIT is set, so submission CSV was skipped because it would not contain every id."
fi
