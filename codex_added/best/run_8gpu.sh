#!/usr/bin/env bash
# Run the best-known pipeline as independent row shards across visible GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_PATH="${1:-data/private.jsonl}"
OUT_DIR="${2:-codex_added/results/best_private_8gpu}"
SUBMISSION_PATH="${3:-codex_added/submissions/best_submission.csv}"

cd "${REPO_ROOT}"
codex_added/scripts/20_run_best_pipeline_8gpu.sh "${DATA_PATH}" "${OUT_DIR}" "${SUBMISSION_PATH}"
