#!/usr/bin/env bash
# Added by Codex: 2-GPU constrained MCQ audit using V6 traces as context.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CODEX_ROOT}/.." && pwd)"
ARCHIVE2_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_PATH="${DATA_PATH:-codex_added/job_data/public200_first100_plus_diagnostic100.jsonl}"
BASELINE_PATH="${BASELINE_PATH:-codex_added/results/public200_strict_audit_2gpu_20260528_092015/v6_slot_aware_finalized_merged.jsonl}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-codex_added/results/public200_mcq_v6_trace_audit_2gpu_${RUN_STAMP}}"
MCQ_DATA="${OUT_DIR}/public200_mcq.jsonl"
MCQ_LIMIT="${MCQ_LIMIT:-}"
GPUS="${GPUS:-0,1}"
MIN_CONSENSUS="${MIN_CONSENSUS:-3}"
AUDIT_VARIANTS="${AUDIT_VARIANTS:-trace_audit,trace_skeptical,trace_final_match,trace_independent}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/.hf-cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-vllm/bin/python}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-4}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.88}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
TRACE_HEAD_CHARS="${TRACE_HEAD_CHARS:-1800}"
TRACE_TAIL_CHARS="${TRACE_TAIL_CHARS:-4200}"

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
        echo "GPUS must contain exactly two GPU ids, got: ${gpus_csv}" >&2
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
            echo "Duplicate GPU id in GPUS=${gpus_csv}." >&2
            exit 1
        fi
        seen+="${gpu_id},"
    done
}

validate_two_gpu_csv "${GPUS}"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/raw" "${TMPDIR}" "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${VLLM_RPC_BASE_PATH}"

exec > >(tee -a "${OUT_DIR}/mcq_v6_trace_audit_2gpu_run.log") 2>&1

echo "run_stamp=${RUN_STAMP}"
echo "data=${DATA_PATH}"
echo "baseline=${BASELINE_PATH}"
echo "out_dir=${OUT_DIR}"
echo "gpus=${GPUS}"
echo "audit_variants=${AUDIT_VARIANTS}"
echo "min_consensus=${MIN_CONSENSUS}"
echo "max_model_len=${MAX_MODEL_LEN}"
echo "model_path=${MODEL_PATH}"
echo "started_at=$(date -u -Iseconds)"

"${PYTHON_BIN}" codex_added/scripts/41_build_mcq_slice.py --data "${DATA_PATH}" --output "${MCQ_DATA}" ${MCQ_LIMIT:+--limit "${MCQ_LIMIT}"}
MCQ_COUNT="$(wc -l < "${MCQ_DATA}" | tr -d ' ')"
SHARD0_OFFSET=0
SHARD0_LIMIT=$(((MCQ_COUNT + 1) / 2))
SHARD1_OFFSET="${SHARD0_LIMIT}"
SHARD1_LIMIT=$((MCQ_COUNT - SHARD1_OFFSET))

echo "mcq_count=${MCQ_COUNT}"
echo "shard_00=gpu:${GPU_IDS[0]},offset:${SHARD0_OFFSET},limit:${SHARD0_LIMIT}"
echo "shard_01=gpu:${GPU_IDS[1]},offset:${SHARD1_OFFSET},limit:${SHARD1_LIMIT}"

run_shard() {
    local shard_name="$1"
    local gpu="$2"
    local offset="$3"
    local limit="$4"
    local output_path="${OUT_DIR}/raw/shard_${shard_name}.jsonl"
    local log_path="${OUT_DIR}/logs/mcq_v6_trace_audit_shard_${shard_name}.log"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        "${PYTHON_BIN}" codex_added/scripts/48_run_mcq_v6_trace_audit.py \
            --data "${MCQ_DATA}" \
            --baseline "${BASELINE_PATH}" \
            --output "${output_path}" \
            --offset "${offset}" \
            --limit "${limit}" \
            --variants "${AUDIT_VARIANTS}" \
            --trace-head-chars "${TRACE_HEAD_CHARS}" \
            --trace-tail-chars "${TRACE_TAIL_CHARS}" \
            --model-id "${MODEL_PATH}" \
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
    echo "MCQ V6 trace audit generation failed. Check ${OUT_DIR}/logs/." >&2
    exit 1
fi

MERGE_OUT="${OUT_DIR}/raw_mcq_v6_trace_audit_merged.jsonl"
MERGE_OUT="${MERGE_OUT}" MCQ_DATA="${MCQ_DATA}" AUDIT_VARIANTS="${AUDIT_VARIANTS}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

mcq_data = Path(os.environ["MCQ_DATA"])
merge_out = Path(os.environ["MERGE_OUT"])
variants = [name.strip() for name in os.environ["AUDIT_VARIANTS"].split(",") if name.strip()]
expected_per_id = len(variants)
shards = [merge_out.parent / "raw" / "shard_00.jsonl", merge_out.parent / "raw" / "shard_01.jsonl"]
expected_ids = [int(json.loads(line)["id"]) for line in mcq_data.read_text(encoding="utf-8").splitlines() if line.strip()]
expected = set(expected_ids)
rows_by_id = defaultdict(list)
for shard in shards:
    with shard.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = int(row["id"])
            if item_id not in expected:
                raise ValueError(f"{shard} contains unexpected id {item_id}")
            row["id"] = item_id
            row["source_shard"] = shard.name
            row["source_shard_row_index"] = index
            rows_by_id[item_id].append(row)
missing = [item_id for item_id in expected_ids if item_id not in rows_by_id]
if missing:
    raise ValueError(f"Missing ids: {missing[:10]}")
bad = {item_id: len(rows_by_id[item_id]) for item_id in expected_ids if len(rows_by_id[item_id]) != expected_per_id}
if bad:
    raise ValueError(f"Expected {expected_per_id} rows per id, mismatches: {dict(list(bad.items())[:10])}")
with merge_out.open("w", encoding="utf-8") as handle:
    total = 0
    for item_id in expected_ids:
        for row in rows_by_id[item_id]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
print(f"Merged {total} MCQ audit rows for {len(expected_ids)} ids into {merge_out}")
PY

"${PYTHON_BIN}" codex_added/scripts/25_analyze_candidate_pool.py \
    --data "${MCQ_DATA}" \
    --responses "${MERGE_OUT}" \
    --labels mcq_v6_trace_audit \
    --output "${OUT_DIR}/candidate_diagnostics.jsonl" \
    --summary-output "${OUT_DIR}/candidate_summary.json"

"${PYTHON_BIN}" codex_added/scripts/42_select_mcq_bestof.py \
    --data "${MCQ_DATA}" \
    --responses "${MERGE_OUT}" \
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

"${PYTHON_BIN}" codex_added/scripts/46_score_mcq_direct_hybrid.py --run-dir "${OUT_DIR}" > "${OUT_DIR}/hybrid_summary.json"

{
    echo "run_stamp=${RUN_STAMP}"
    echo "data=${DATA_PATH}"
    echo "baseline=${BASELINE_PATH}"
    echo "out_dir=${OUT_DIR}"
    echo "audit_variants=${AUDIT_VARIANTS}"
    echo "min_consensus=${MIN_CONSENSUS}"
    echo "raw_candidates=${MERGE_OUT}"
    echo "hybrid=${OUT_DIR}/hybrid_v6_mcq_direct_consensus.jsonl"
    echo "completed_at=$(date -u -Iseconds)"
} > "${OUT_DIR}/RUN_INFO.txt"

echo "Done."
echo "Candidate summary: ${OUT_DIR}/candidate_summary.json"
echo "Selection summary: ${OUT_DIR}/selection_summary.json"
echo "Hybrid summary: ${OUT_DIR}/hybrid_summary.json"
