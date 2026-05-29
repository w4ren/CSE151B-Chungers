# Inference README

This note is for Codex instances running inference in this archived repo.
Keep all edits and generated files inside `archive2/CSE151B-Chungers-wenhao-lfs`.

## Recommended public100 vLLM run

Use the vLLM pipeline, not the Transformers fallback:

```bash
cd /home/mnt/BiomechAI/Wenhao/archive2/CSE151B-Chungers-wenhao-lfs

RUN_SMOKE=1 \
GPUS=0,1 \
EXPECTED_GPUS=2 \
REASON_VLLM_BATCH_SIZE=1 \
REASON_MAX_NUM_SEQS=2 \
MAX_NEW_TOKENS=8192 \
codex_added/scripts/33_run_public100_8k_full_pipeline_2gpu.sh
```

Before launching, run `nvidia-smi`. The intended two-GPU state is that both
selected GPUs are mostly free. Never launch with duplicate ids such as
`GPUS=1,1`; the scripts now refuse duplicate GPU ids because that stacks both
shards on one GPU. The scripts also refuse to start if a selected GPU has more
than `MAX_GPU_USED_MIB=1000` MiB already allocated; override that only when the
contention is intentional.

## Why these defaults

- `REASON_VLLM_BATCH_SIZE=1` writes every completed raw problem to JSONL. With
  `0`, vLLM processes the whole shard as one chunk and writes only after the
  entire chunk returns. A run can reach `49/50` in the progress log and still
  save `0` rows if the last prompt never returns.
- `REASON_MAX_NUM_SEQS=2` gives vLLM some scheduler concurrency on A10s without
  being too aggressive.
- `MAX_NEW_TOKENS=8192` is intentional for the raw reasoning pass. The speed
  fixes here are GPU placement, incremental flushing, vLLM scheduler settings,
  and avoiding slow defaults like forced BNB/eager mode.
- `VLLM_QUANTIZATION=none` and `ENFORCE_EAGER=0` are faster defaults. If the
  model OOMs, retry with:

```bash
VLLM_QUANTIZATION=bitsandbytes VLLM_LOAD_FORMAT=bitsandbytes
```

Use `ENFORCE_EAGER=1` only if CUDA graph or compile behavior is unstable.

## Rerunning only a missing shard

If the first 50 rows are already saved and only the second half is missing,
rerun only `offset=50, limit=50`. For a one-GPU rerun:

```bash
cd /home/mnt/BiomechAI/Wenhao/archive2/CSE151B-Chungers-wenhao-lfs

RUN_SMOKE=0 \
ROW_OFFSET=50 \
ROW_LIMIT=50 \
GPUS=0 \
EXPECTED_GPUS=1 \
REASON_VLLM_BATCH_SIZE=1 \
REASON_MAX_NUM_SEQS=2 \
MAX_NEW_TOKENS=8192 \
OUT_DIR=codex_added/results/public100_shard01_rerun_$(date -u +%Y%m%d_%H%M%S) \
codex_added/scripts/33_run_public100_8k_full_pipeline_2gpu.sh
```

For a two-GPU rerun of the missing half, use `GPUS=0,1 EXPECTED_GPUS=2`; the
script will split those 50 rows across two shards.

## Monitoring

Check GPU placement after launch:

```bash
nvidia-smi
```

Expected two-GPU behavior: one Python/vLLM process on each selected GPU during
the raw stage. If GPU 0 is idle and GPU 1 has both processes, stop the run and
restart with `GPUS=0,1`.

Check saved raw progress:

```bash
wc -l codex_added/results/<run>/raw/*.jsonl
tail -80 codex_added/results/<run>/logs/reason_8k_shard_*.log
```

Progress bars in logs are not saved predictions. The JSONL row count is the
source of truth.

## Script map

- `codex_added/scripts/33_run_public100_8k_full_pipeline_2gpu.sh`: preferred
  vLLM pipeline.
- `codex_added/scripts/34_run_public100_8k_full_pipeline_transformers_2gpu.sh`:
  slower Transformers fallback. Use only when vLLM is broken.
- `codex_added/scripts/10_run_prompt_sweep.py`: raw reasoning writer. vLLM
  writes after each configured chunk, so keep `--vllm-batch-size` small.
