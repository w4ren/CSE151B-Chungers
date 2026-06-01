#!/usr/bin/env bash
# Added by Codex: periodically copy and fsync live raw shard outputs.

set -euo pipefail

RUN_DIR="${1:?usage: 35_checkpoint_live_raw.sh RUN_DIR [INTERVAL_SECONDS]}"
INTERVAL_SECONDS="${2:-30}"
SNAP_ROOT="${RUN_DIR}/snapshots"
SNAP_DIR="${SNAP_ROOT}/live_latest"
LOG_PATH="${SNAP_ROOT}/live_checkpoint.log"

mkdir -p "${SNAP_DIR}"

while true; do
  tmp_dir="${SNAP_ROOT}/live_latest.tmp.$$"
  mkdir -p "${tmp_dir}"
  copied=0

  for shard_path in "${RUN_DIR}"/raw/shard_*.jsonl; do
    if [[ -f "${shard_path}" ]]; then
      cp "${shard_path}" "${tmp_dir}/$(basename "${shard_path}")"
      copied=1
    fi
  done

  if [[ "${copied}" -eq 1 ]]; then
    for tmp_file in "${tmp_dir}"/*.jsonl; do
      if [[ -f "${tmp_file}" ]]; then
        dest="${SNAP_DIR}/$(basename "${tmp_file}")"
        mv -f "${tmp_file}" "${dest}"
        sync -f "${dest}" || true
      fi
    done
    for shard_path in "${RUN_DIR}"/raw/shard_*.jsonl; do
      if [[ -f "${shard_path}" ]]; then
        sync -f "${shard_path}" || true
      fi
    done
    {
      printf '%s\n' "$(date -u -Iseconds)"
      for shard_path in "${SNAP_DIR}"/shard_*.jsonl; do
        if [[ -f "${shard_path}" ]]; then
          printf '%s lines=%s\n' "${shard_path}" "$(wc -l < "${shard_path}")"
        fi
      done
    } >> "${LOG_PATH}"
    sync -f "${LOG_PATH}" || true
  fi

  rm -rf "${tmp_dir}"
  if [[ -f "${RUN_DIR}/RUN_INFO.txt" ]]; then
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done
