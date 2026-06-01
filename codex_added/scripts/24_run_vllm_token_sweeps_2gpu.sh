#!/usr/bin/env bash
# Added by Codex: two-GPU vLLM token-budget sweeps; not part of the original starter repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_PATH="${1:-data/public.jsonl}"
OUT_DIR="${2:-codex_added/results/vllm_30_public_4k_8k_16k}"
OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-30}"
GPUS="${GPUS:-0,1}"
BUDGETS="${BUDGETS:-4096 8192 16384}"
PROMPT_TOKEN_ALLOWANCE="${PROMPT_TOKEN_ALLOWANCE:-2048}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-0}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-bitsandbytes}"
VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-bitsandbytes}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv-vllm/bin/python" ]]; then
    PYTHON=".venv-vllm/bin/python"
  else
    PYTHON="python"
  fi
fi

if [[ -z "${MODEL_ID:-}" ]]; then
  SNAPSHOT_DIR="$(find .hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null || true)"
  MODEL_ID="${SNAPSHOT_DIR:-Qwen/Qwen3-4B-Thinking-2507}"
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${REPO_ROOT}/.vllm-cache}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing data file: ${DATA_PATH}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "No GPU ids configured. Set GPUS=0,1." >&2
  exit 1
fi

TOTAL_ROWS="$(wc -l < "${DATA_PATH}")"
RUN_ROWS="${LIMIT}"
if [[ "${RUN_ROWS}" -le 0 ]]; then
  echo "LIMIT must be positive." >&2
  exit 1
fi
if [[ $((OFFSET + RUN_ROWS)) -gt "${TOTAL_ROWS}" ]]; then
  echo "OFFSET + LIMIT exceeds dataset rows: ${OFFSET} + ${RUN_ROWS} > ${TOTAL_ROWS}" >&2
  exit 1
fi

NUM_SHARDS="${NUM_SHARDS:-${#GPU_IDS[@]}}"
if [[ "${NUM_SHARDS}" -gt "${RUN_ROWS}" ]]; then
  NUM_SHARDS="${RUN_ROWS}"
fi

mkdir -p "${OUT_DIR}/logs"

echo "Data: ${DATA_PATH}"
echo "Rows: [${OFFSET}, $((OFFSET + RUN_ROWS)))"
echo "Budgets: ${BUDGETS}"
echo "GPUs: ${GPUS}"
echo "Python: ${PYTHON}"
echo "Model: ${MODEL_ID}"
echo "Quantization: ${VLLM_QUANTIZATION:-none}"
echo "Output: ${OUT_DIR}"

merged_files=()
merged_labels=()

for budget in ${BUDGETS}; do
  budget_dir="${OUT_DIR}/${budget}"
  mkdir -p "${budget_dir}"
  max_model_len=$((budget + PROMPT_TOKEN_ALLOWANCE))
  max_num_batched_tokens="${max_model_len}"

  base_size=$((RUN_ROWS / NUM_SHARDS))
  remainder=$((RUN_ROWS % NUM_SHARDS))
  shard_offset="${OFFSET}"
  pids=()
  labels=()
  shard_files=()

  echo "Starting ${budget}-token sweep with max_model_len=${max_model_len}"
  for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    count="${base_size}"
    if [[ "${shard}" -lt "${remainder}" ]]; then
      count=$((count + 1))
    fi

    label="$(printf "%02d" "${shard}")"
    gpu="${GPU_IDS[$((shard % ${#GPU_IDS[@]}))]}"
    output_path="${budget_dir}/shard_${label}.jsonl"
    log_path="${OUT_DIR}/logs/${budget}_shard_${label}.log"
    shard_files+=("${output_path}")

    echo "  shard ${label}: GPU ${gpu}, OFFSET=${shard_offset}, LIMIT=${count}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      extra_args=()
      if [[ -n "${VLLM_QUANTIZATION}" ]]; then
        extra_args+=(--vllm-quantization "${VLLM_QUANTIZATION}" --vllm-load-format "${VLLM_LOAD_FORMAT}")
      fi
      if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
        extra_args+=(--enforce-eager)
      fi
      "${PYTHON}" codex_added/scripts/10_run_prompt_sweep.py \
        --input "${DATA_PATH}" \
        --output "${output_path}" \
        --offset "${shard_offset}" \
        --limit "${count}" \
        --variants final_box_only \
        --num-samples 1 \
        --do-sample \
        --quantization none \
        --max-new-tokens "${budget}" \
        --answer-key-mode strict \
        --backend vllm \
        --model-id "${MODEL_ID}" \
        --dtype "${VLLM_DTYPE}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${max_model_len}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${max_num_batched_tokens}" \
        --vllm-batch-size "${VLLM_BATCH_SIZE}" \
        "${extra_args[@]}"
    ) > "${log_path}" 2>&1 &

    pids+=("$!")
    labels+=("${label}")
    shard_offset=$((shard_offset + count))
  done

  failed=0
  for index in "${!pids[@]}"; do
    pid="${pids[$index]}"
    label="${labels[$index]}"
    if wait "${pid}"; then
      echo "  shard ${label} finished."
    else
      echo "  shard ${label} failed; see ${OUT_DIR}/logs/${budget}_shard_${label}.log" >&2
      failed=1
    fi
  done

  if [[ "${failed}" != "0" ]]; then
    exit 1
  fi

  merged_path="${budget_dir}/merged.jsonl"
  "${PYTHON}" codex_added/scripts/21_merge_shard_predictions.py \
    --data "${DATA_PATH}" \
    --offset "${OFFSET}" \
    --limit "${RUN_ROWS}" \
    --predictions "${shard_files[@]}" \
    --output "${merged_path}"
  merged_files+=("${merged_path}")
  merged_labels+=("${budget}")
  echo "Finished ${budget}-token sweep: ${merged_path}"
done

if [[ "${RUN_POSTPROCESS:-1}" == "1" ]]; then
  postprocess_score_args=()
  if [[ "${POSTPROCESS_SCORE:-1}" == "1" ]]; then
    postprocess_score_args=(--score)
  fi

  "${PYTHON}" codex_added/scripts/25_analyze_candidate_pool.py \
    --data "${DATA_PATH}" \
    --offset "${OFFSET}" \
    --limit "${RUN_ROWS}" \
    --responses "${merged_files[@]}" \
    --labels "${merged_labels[@]}" \
    --output "${OUT_DIR}/candidate_diagnostics.jsonl" \
    --summary-output "${OUT_DIR}/candidate_summary.json"

  "${PYTHON}" codex_added/scripts/26_select_budget_candidates.py \
    --data "${DATA_PATH}" \
    --offset "${OFFSET}" \
    --limit "${RUN_ROWS}" \
    --responses "${merged_files[@]}" \
    --labels "${merged_labels[@]}" \
    --output "${OUT_DIR}/selected_consensus.jsonl" \
    "${postprocess_score_args[@]}"
fi

echo "All sweeps finished under ${OUT_DIR}"
