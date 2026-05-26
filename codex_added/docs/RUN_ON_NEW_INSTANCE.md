<!-- Added by Codex: runbook for reproducing this workflow on a fresh GPU instance. -->

# Run On A New Instance

This guide assumes the repository has already been copied or cloned to the new
machine. The original starter code lives at the repository root. The reusable
competition workflow lives under `codex_added/`.

## What You Need

- Linux machine with an NVIDIA GPU.
- Python 3.10 or 3.11.
- Enough GPU memory for `Qwen/Qwen3-4B-Thinking-2507`.
  - The previous working setup used an NVIDIA A30 with 24GB VRAM.
  - The current best scripts use unquantized Transformers inference, so 11GB
    class GPUs are usually too tight for the full method.
- The competition private file at `data/private.jsonl` for final submission.
- The answer-format LoRA adapter at:

```text
codex_added/models/qwen3_answer_format_lora
```

`codex_added/models/` is ignored by Git because adapters are large. On a new
instance, either copy that adapter directory from the old instance or rebuild it
with the training command below.

## Setup

From the repository root:

```bash
cd /path/to/151B_SP26_Competition
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r codex_added/requirements.txt
```

If the new instance already has a working CUDA-enabled PyTorch build, keep it.
The requirements file intentionally does not pin `torch` because cluster images
often provide their own compatible build.

Verify the basic environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -c "import transformers; print(transformers.__version__)"
```

## Data Placement

The `data/` directory is ignored by Git, so a clone may not include any
competition JSONL files. Copy them from the competition bundle or previous
instance.

Public data is expected at:

```text
data/public.jsonl
```

Private competition data must be placed at:

```text
data/private.jsonl
```

The final CSV writer uses `data/private.jsonl` for id ordering, so the private
file must be the exact competition file.

## Adapter Setup

Preferred: copy the trained adapter from the previous instance:

```text
codex_added/models/qwen3_answer_format_lora/
```

Check that it exists:

```bash
test -d codex_added/models/qwen3_answer_format_lora && echo "adapter found"
```

If you need to rebuild it, run:

```bash
python codex_added/scripts/12_build_answer_format_sft.py \
  --public data/public.jsonl \
  --output codex_added/data/sft_answer_format.jsonl

python codex_added/scripts/13_train_lora_sft.py \
  --train codex_added/data/sft_answer_format.jsonl \
  --output-dir codex_added/models/qwen3_answer_format_lora \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 1536
```

On the previous A30 setup, this took about 6 minutes.

## Smoke Tests

Inspect the public dataset:

```bash
python codex_added/scripts/analyze_public.py --data data/public.jsonl
```

Run a tiny public inference test:

```bash
python codex_added/scripts/run_inference.py \
  --config codex_added/configs/baseline.yaml \
  --data data/public.jsonl \
  --output codex_added/results/public_smoke.jsonl \
  --limit 2
```

Run the current best pipeline on a small public slice:

```bash
SCORE=1 LIMIT=5 \
  codex_added/best/run.sh \
  data/public.jsonl \
  codex_added/results/best_public_smoke \
  codex_added/submissions/best_public_smoke.csv
```

When `LIMIT` is set, the script skips CSV creation because the output is only a
partial dataset.

## Full Private Submission

Single process:

```bash
codex_added/best/run.sh \
  data/private.jsonl \
  codex_added/results/best_private \
  codex_added/submissions/best_submission.csv
```

Multi-GPU on one visible machine:

```bash
GPUS=0,1,2,3,4,5,6,7 NUM_SHARDS=8 \
  codex_added/best/run_8gpu.sh \
  data/private.jsonl \
  codex_added/results/best_private_8gpu \
  codex_added/submissions/best_submission.csv
```

For a 4-GPU job, use:

```bash
GPUS=0,1,2,3 NUM_SHARDS=4 \
  codex_added/best/run_8gpu.sh \
  data/private.jsonl \
  codex_added/results/best_private_4gpu \
  codex_added/submissions/best_submission.csv
```

The final submission is:

```text
codex_added/submissions/best_submission.csv
```

## Two-Job Cluster Runs

If the scheduler gives two separate jobs, split the private file first:

```bash
python codex_added/scripts/22_split_jsonl_for_jobs.py \
  --data data/private.jsonl \
  --out-dir codex_added/job_data \
  --prefix private \
  --num-jobs 2
```

Run one command in each job. If each job sees its allocated GPUs as `0,1,2,3`,
use this shape for both jobs, changing only the input and output paths:

```bash
GPUS=0,1,2,3 NUM_SHARDS=4 WRITE_SUBMISSION=0 \
  codex_added/scripts/20_run_best_pipeline_8gpu.sh \
  codex_added/job_data/private_job0.jsonl \
  codex_added/results/best_private_job0 \
  codex_added/submissions/best_submission_job0_unused.csv

GPUS=0,1,2,3 NUM_SHARDS=4 WRITE_SUBMISSION=0 \
  codex_added/scripts/20_run_best_pipeline_8gpu.sh \
  codex_added/job_data/private_job1.jsonl \
  codex_added/results/best_private_job1 \
  codex_added/submissions/best_submission_job1_unused.csv
```

Merge the selected predictions and write the final CSV:

```bash
python codex_added/scripts/21_merge_shard_predictions.py \
  --data data/private.jsonl \
  --output codex_added/results/best_private_merged/selected.jsonl \
  --predictions \
  codex_added/results/best_private_job0/selected.jsonl \
  codex_added/results/best_private_job1/selected.jsonl

python codex_added/scripts/make_submission.py \
  --data data/private.jsonl \
  --predictions codex_added/results/best_private_merged/selected.jsonl \
  --output codex_added/submissions/best_submission.csv
```

## Optional vLLM Setup

vLLM can speed up batched generation on supported GPUs, but the current best
pipeline still uses Transformers scripts. Installing vLLM alone will not speed
up `codex_added/best/run.sh` until those scripts are ported.

Use a separate environment for vLLM experiments because vLLM has strict
Torch/CUDA dependencies:

```bash
python -m pip install -U uv
uv venv .venv-vllm --python 3.11 --seed
source .venv-vllm/bin/activate
uv pip install -r codex_added/requirements.txt
uv pip install "vllm>=0.8.5" --torch-backend=auto
```

Verify:

```bash
python -c "import torch, vllm; print(vllm.__version__); print(torch.__version__); print(torch.cuda.get_device_name(0))"
```

Smoke test the existing vLLM-capable baseline path:

```bash
python codex_added/scripts/run_inference.py \
  --config codex_added/configs/baseline.yaml \
  --backend vllm \
  --data data/public.jsonl \
  --output codex_added/results/vllm_smoke.jsonl \
  --limit 2
```

## Troubleshooting

- `Missing data file: data/private.jsonl`: place the private competition JSONL
  under `data/private.jsonl`.
- `Missing LoRA adapter directory`: copy or rebuild
  `codex_added/models/qwen3_answer_format_lora`.
- CUDA out of memory during the best pipeline: reduce shard concurrency, use a
  larger GPU, or run fewer GPUs per node. The current scripts load the model in
  separate processes.
- `bitsandbytes` or quantization errors: start with the documented
  `--quantization none` commands. The best pipeline already uses
  `--quantization none`.
- vLLM import or kernel errors: use a clean vLLM environment and a supported GPU.
  Do not debug vLLM inside the same environment needed for the Transformers
  submission path unless you are ready to repair Torch/CUDA package versions.
