#!/usr/bin/env bash
# Run the best-known 13/20 selection pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_PATH="${1:-data/private.jsonl}"
OUT_DIR="${2:-codex_added/results/best_private}"
SUBMISSION_PATH="${3:-codex_added/submissions/best_submission.csv}"

cd "${REPO_ROOT}"
codex_added/scripts/19_run_best_pipeline.sh "${DATA_PATH}" "${OUT_DIR}" "${SUBMISSION_PATH}"
