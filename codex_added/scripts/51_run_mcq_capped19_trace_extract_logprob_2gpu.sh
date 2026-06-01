#!/usr/bin/env bash
# Added by Codex: targeted 2-GPU MCQ extraction/logprob test on V6 capped-wrong MCQs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CODEX_ROOT}/.." && pwd)"
ARCHIVE2_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_PATH="${DATA_PATH:-codex_added/job_data/public200_first100_plus_diagnostic100.jsonl}"
RAW_PATH="${RAW_PATH:-codex_added/results/public200_strict_audit_2gpu_20260528_092015/raw_8k_merged.jsonl}"
BASELINE_PATH="${BASELINE_PATH:-codex_added/results/public200_strict_audit_2gpu_20260528_092015/v6_slot_aware_finalized_merged.jsonl}"
CURRENT_FINALIZED_PATH="${CURRENT_FINALIZED_PATH:-codex_added/results/public200_strict_audit_2gpu_20260528_092015/finalized_8k_lora_merged.jsonl}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-codex_added/results/mcq_capped19_trace_extract_logprob_2gpu_${RUN_STAMP}}"
GPUS="${GPUS:-0,1}"
METHODS="${METHODS:-extractor_letter,logprob}"
TARGET_IDS="${TARGET_IDS:-}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/.hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-vllm/bin/python}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TRACE_CHARS="${MAX_TRACE_CHARS:-18000}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.88}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-2}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"

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
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

validate_two_gpu_csv() {
    local gpus_csv="$1"
    local gpu_ids=()
    IFS=',' read -r -a gpu_ids <<< "${gpus_csv}"
    if [[ "${#gpu_ids[@]}" -ne 2 ]]; then
        echo "GPUS must contain exactly two GPU ids, got ${gpus_csv}" >&2
        exit 1
    fi
    local seen=","
    local gpu_id
    for gpu_id in "${gpu_ids[@]}"; do
        if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
            echo "Invalid GPU id: ${gpu_id}" >&2
            exit 1
        fi
        if [[ "${seen}" == *",${gpu_id},"* ]]; then
            echo "Duplicate GPU id in GPUS=${gpus_csv}" >&2
            exit 1
        fi
        seen+="${gpu_id},"
    done
}

validate_two_gpu_csv "${GPUS}"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/raw" "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${VLLM_RPC_BASE_PATH}"
exec > >(tee -a "${OUT_DIR}/mcq_capped19_trace_extract_logprob_2gpu.log") 2>&1

for file in "${DATA_PATH}" "${RAW_PATH}" "${BASELINE_PATH}" "${CURRENT_FINALIZED_PATH}"; do
    if [[ ! -f "${file}" ]]; then
        echo "Missing required file: ${file}" >&2
        exit 1
    fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python environment: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing local model snapshot: ${MODEL_PATH}" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
fi

echo "run_stamp=${RUN_STAMP}"
echo "out_dir=${OUT_DIR}"
echo "data=${DATA_PATH}"
echo "raw=${RAW_PATH}"
echo "baseline=${BASELINE_PATH}"
echo "current_finalized=${CURRENT_FINALIZED_PATH}"
echo "gpus=${GPUS}"
echo "methods=${METHODS}"
if [[ -n "${TARGET_IDS}" ]]; then
    echo "target_ids=${TARGET_IDS}"
fi
echo "max_model_len=${MAX_MODEL_LEN}"
echo "max_trace_chars=${MAX_TRACE_CHARS}"
echo "started_at=$(date -u -Iseconds)"

TARGET_ARGS=()
if [[ -n "${TARGET_IDS}" ]]; then
    TARGET_ARGS+=(--target-ids "${TARGET_IDS}")
fi

"${PYTHON_BIN}" codex_added/scripts/50_run_mcq_trace_extract_logprob.py \
    --data "${DATA_PATH}" \
    --raw-responses "${RAW_PATH}" \
    --baseline "${BASELINE_PATH}" \
    --current-finalized "${CURRENT_FINALIZED_PATH}" \
    "${TARGET_ARGS[@]}" \
    --prepare-only \
    --prepare-target-data "${OUT_DIR}/target_data.jsonl" \
    --prepare-target-raw "${OUT_DIR}/target_raw_8k.jsonl" \
    --summary-output "${OUT_DIR}/prepare_summary.json"

TARGET_COUNT="$(wc -l < "${OUT_DIR}/target_data.jsonl" | tr -d ' ')"
SHARD0_OFFSET=0
SHARD0_LIMIT=$(((TARGET_COUNT + 1) / 2))
SHARD1_OFFSET="${SHARD0_LIMIT}"
SHARD1_LIMIT=$((TARGET_COUNT - SHARD1_OFFSET))

echo "target_count=${TARGET_COUNT}"
echo "shard_00=gpu:${GPU_IDS[0]},offset:${SHARD0_OFFSET},limit:${SHARD0_LIMIT}"
echo "shard_01=gpu:${GPU_IDS[1]},offset:${SHARD1_OFFSET},limit:${SHARD1_LIMIT}"

run_shard() {
    local shard_name="$1"
    local gpu="$2"
    local offset="$3"
    local limit="$4"
    local output_path="${OUT_DIR}/raw/shard_${shard_name}.jsonl"
    local log_path="${OUT_DIR}/logs/shard_${shard_name}.log"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PYTHON_BIN}" codex_added/scripts/50_run_mcq_trace_extract_logprob.py \
            --data "${DATA_PATH}" \
            --raw-responses "${RAW_PATH}" \
            --baseline "${BASELINE_PATH}" \
            --current-finalized "${CURRENT_FINALIZED_PATH}" \
            "${TARGET_ARGS[@]}" \
            --output "${output_path}" \
            --offset "${offset}" \
            --limit "${limit}" \
            --methods "${METHODS}" \
            --max-trace-chars "${MAX_TRACE_CHARS}" \
            --model-id "${MODEL_PATH}" \
            --backend vllm \
            --dtype float16 \
            --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
            --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
            --vllm-batch-size "${VLLM_BATCH_SIZE}" \
            --vllm-quantization bitsandbytes \
            --vllm-load-format bitsandbytes \
            --attention-backend "${VLLM_ATTENTION_BACKEND}" \
            --enforce-eager \
            --safetensors-load-strategy prefetch \
            --disable-async-scheduling
    ) > "${log_path}" 2>&1 &
    LAST_SHARD_PID="$!"
}

pids=()
run_shard "00" "${GPU_IDS[0]}" "${SHARD0_OFFSET}" "${SHARD0_LIMIT}"
pids+=("${LAST_SHARD_PID}")
run_shard "01" "${GPU_IDS[1]}" "${SHARD1_OFFSET}" "${SHARD1_LIMIT}"
pids+=("${LAST_SHARD_PID}")
echo "started shard pids: ${pids[*]}"

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if [[ "${failed}" -ne 0 ]]; then
    echo "Targeted MCQ extraction/logprob run failed. Check ${OUT_DIR}/logs." >&2
    exit 1
fi

"${PYTHON_BIN}" codex_added/scripts/50_run_mcq_trace_extract_logprob.py \
    --data "${DATA_PATH}" \
    --raw-responses "${RAW_PATH}" \
    --baseline "${BASELINE_PATH}" \
    --current-finalized "${CURRENT_FINALIZED_PATH}" \
    "${TARGET_ARGS[@]}" \
    --predictions "${OUT_DIR}/raw/shard_00.jsonl" "${OUT_DIR}/raw/shard_01.jsonl" \
    --summary-output "${OUT_DIR}/summary.json" \
    --details-output "${OUT_DIR}/details.jsonl"

cat > "${OUT_DIR}/RUN_INFO.txt" <<EOF
run_stamp=${RUN_STAMP}
out_dir=${OUT_DIR}
data=${DATA_PATH}
raw=${RAW_PATH}
baseline=${BASELINE_PATH}
current_finalized=${CURRENT_FINALIZED_PATH}
gpus=${GPUS}
methods=${METHODS}
target_ids=${TARGET_IDS}
max_model_len=${MAX_MODEL_LEN}
max_trace_chars=${MAX_TRACE_CHARS}
summary=${OUT_DIR}/summary.json
details=${OUT_DIR}/details.jsonl
completed_at=$(date -u -Iseconds)
EOF

echo "completed_at=$(date -u -Iseconds)"
echo "summary=${OUT_DIR}/summary.json"
