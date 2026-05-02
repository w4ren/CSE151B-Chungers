#!/usr/bin/env bash
# Added by Codex: run the best-known pipeline as independent shards across GPUs.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  codex_added/scripts/20_run_best_pipeline_8gpu.sh DATA_PATH OUT_DIR SUBMISSION_PATH

Environment:
  GPUS        Comma-separated GPU ids. Default: 0,1,2,3,4,5,6,7
  NUM_SHARDS  Number of row shards to launch. Default: number of GPUS
  ROW_OFFSET  Dataset row offset for this job. Default: 0
  ROW_LIMIT   Number of dataset rows for this job. Default: all rows after ROW_OFFSET
  ADAPTER_DIR LoRA adapter directory. Default: codex_added/models/qwen3_answer_format_lora
  SCORE       Set SCORE=1 to score public/labeled data.
  WRITE_SUBMISSION
              Set WRITE_SUBMISSION=0 for partial/split-data jobs that will be merged later.

Examples:
  codex_added/scripts/20_run_best_pipeline_8gpu.sh data/private.jsonl codex_added/results/best_private_8gpu codex_added/submissions/best_submission.csv

  GPUS=0,1,2,3 NUM_SHARDS=4 WRITE_SUBMISSION=0 \
    codex_added/scripts/20_run_best_pipeline_8gpu.sh codex_added/job_data/private_job0.jsonl codex_added/results/best_private_job0
EOF
  exit 0
fi

DATA_PATH="${1:-data/private.jsonl}"
OUT_DIR="${2:-codex_added/results/best_private_8gpu}"
SUBMISSION_PATH="${3:-codex_added/submissions/best_submission.csv}"
ADAPTER_DIR="${ADAPTER_DIR:-codex_added/models/qwen3_answer_format_lora}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SCORE="${SCORE:-0}"
ROW_OFFSET="${ROW_OFFSET:-0}"
ROW_LIMIT="${ROW_LIMIT:-}"
WRITE_SUBMISSION="${WRITE_SUBMISSION:-1}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing data file: ${DATA_PATH}" >&2
  echo "Usage: $0 data/private.jsonl codex_added/results/best_private_8gpu codex_added/submissions/best_submission.csv" >&2
  exit 1
fi

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "Missing LoRA adapter directory: ${ADAPTER_DIR}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "No GPU ids configured. Set GPUS=0,1,2,3,4,5,6,7." >&2
  exit 1
fi

NUM_SHARDS="${NUM_SHARDS:-${#GPU_IDS[@]}}"
if [[ "${NUM_SHARDS}" -le 0 ]]; then
  echo "NUM_SHARDS must be positive." >&2
  exit 1
fi

TOTAL_ROWS="$(wc -l < "${DATA_PATH}")"
if [[ "${TOTAL_ROWS}" -le 0 ]]; then
  echo "No rows found in ${DATA_PATH}." >&2
  exit 1
fi

if [[ "${ROW_OFFSET}" -lt 0 ]]; then
  echo "ROW_OFFSET must be non-negative." >&2
  exit 1
fi
if [[ "${ROW_OFFSET}" -ge "${TOTAL_ROWS}" ]]; then
  echo "ROW_OFFSET=${ROW_OFFSET} is outside ${TOTAL_ROWS} rows." >&2
  exit 1
fi

if [[ -n "${ROW_LIMIT}" ]]; then
  if [[ "${ROW_LIMIT}" -le 0 ]]; then
    echo "ROW_LIMIT must be positive when set." >&2
    exit 1
  fi
  RUN_ROWS="${ROW_LIMIT}"
else
  RUN_ROWS=$((TOTAL_ROWS - ROW_OFFSET))
fi

if [[ $((ROW_OFFSET + RUN_ROWS)) -gt "${TOTAL_ROWS}" ]]; then
  echo "ROW_OFFSET + ROW_LIMIT exceeds dataset rows: ${ROW_OFFSET} + ${RUN_ROWS} > ${TOTAL_ROWS}" >&2
  exit 1
fi

if [[ "${NUM_SHARDS}" -gt "${RUN_ROWS}" ]]; then
  NUM_SHARDS="${RUN_ROWS}"
fi

mkdir -p "${OUT_DIR}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "Dataset rows: ${TOTAL_ROWS}"
echo "Running rows [${ROW_OFFSET}, $((ROW_OFFSET + RUN_ROWS))) as ${NUM_SHARDS} shards on GPUs: ${GPUS}"
echo "Outputs: ${OUT_DIR}"

base_size=$((RUN_ROWS / NUM_SHARDS))
remainder=$((RUN_ROWS % NUM_SHARDS))
offset="${ROW_OFFSET}"
pids=()
labels=()

for ((shard = 0; shard < NUM_SHARDS; shard++)); do
  count="${base_size}"
  if [[ "${shard}" -lt "${remainder}" ]]; then
    count=$((count + 1))
  fi

  label="$(printf "%02d" "${shard}")"
  gpu="${GPU_IDS[$((shard % ${#GPU_IDS[@]}))]}"
  shard_dir="${OUT_DIR}/shard_${label}"
  log_path="${LOG_DIR}/shard_${label}.log"
  mkdir -p "${shard_dir}"

  echo "Shard ${label}: GPU ${gpu}, OFFSET=${offset}, LIMIT=${count}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export OFFSET="${offset}"
    export LIMIT="${count}"
    export SCORE="${SCORE}"
    export ADAPTER_DIR="${ADAPTER_DIR}"
    codex_added/scripts/19_run_best_pipeline.sh \
      "${DATA_PATH}" \
      "${shard_dir}" \
      "${shard_dir}/partial_submission.csv"
  ) > "${log_path}" 2>&1 &

  pids+=("$!")
  labels+=("${label}")
  offset=$((offset + count))
done

failed=0
for index in "${!pids[@]}"; do
  pid="${pids[$index]}"
  label="${labels[$index]}"
  if wait "${pid}"; then
    echo "Shard ${label} finished."
  else
    echo "Shard ${label} failed. See ${LOG_DIR}/shard_${label}.log" >&2
    failed=1
  fi
done

if [[ "${failed}" != "0" ]]; then
  exit 1
fi

selected_files=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
  label="$(printf "%02d" "${shard}")"
  selected_files+=("${OUT_DIR}/shard_${label}/selected.jsonl")
done

python codex_added/scripts/21_merge_shard_predictions.py \
  --data "${DATA_PATH}" \
  --offset "${ROW_OFFSET}" \
  --limit "${RUN_ROWS}" \
  --output "${OUT_DIR}/selected.jsonl" \
  --predictions "${selected_files[@]}"

if [[ "${WRITE_SUBMISSION}" == "1" && "${ROW_OFFSET}" == "0" && "${RUN_ROWS}" == "${TOTAL_ROWS}" ]]; then
  python codex_added/scripts/make_submission.py \
    --data "${DATA_PATH}" \
    --predictions "${OUT_DIR}/selected.jsonl" \
    --output "${SUBMISSION_PATH}"
  echo "Wrote submission CSV to ${SUBMISSION_PATH}"
else
  echo "Submission CSV skipped; merge job-level selected.jsonl files before writing a full CSV."
fi

echo "Wrote merged JSONL to ${OUT_DIR}/selected.jsonl"
