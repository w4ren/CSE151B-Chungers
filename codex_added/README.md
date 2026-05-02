<!-- Added by Codex: index for Codex-created files; not part of the original starter repository. -->

# Codex-Added Files

This folder collects the files added around the original starter repository.
The starter files at the repo root are unchanged.

## Layout

```text
codex_added/
  best/          Current best 13/20 method, run scripts, K8s manifest, public eval artifacts
  second_best/   Simpler 2048-pass + LoRA finalizer method
  third_best/    Conflict-only rerank + LoRA finalizer method
  archive/       Old generated results, non-current adapters, generated training data
  configs/        Baseline model and generation config
  docs/           Getting-started workflow notes
  math_comp/      Reusable data, prompt, inference, scoring, and submission code
  models/         Current answer-format LoRA adapter
  job_data/       Ignored generated JSONL chunks for multi-job runs
  results/        Fresh run outputs; historical outputs are archived
  scripts/        CLI entry points
  submissions/    Local generated CSV submissions
  requirements.txt
```

## Common Commands

```bash
codex_added/best/run.sh data/private.jsonl codex_added/results/best_private codex_added/submissions/best_submission.csv
codex_added/best/run_8gpu.sh data/private.jsonl codex_added/results/best_private_8gpu codex_added/submissions/best_submission.csv
codex_added/second_best/run.sh data/private.jsonl codex_added/results/second_best_private codex_added/submissions/second_best_submission.csv
codex_added/third_best/run.sh data/private.jsonl codex_added/results/third_best_private codex_added/submissions/third_best_submission.csv
```

Kubernetes manifests for the 2-job, 4-GPU-per-job setup are in `codex_added/best/k8s/cse151b-best-2x4gpu.yaml`.

For two scheduler jobs with four GPUs each, split the data first, then run the same job script/spec shape on each chunk. If the scheduler remaps each job's allocated GPUs to `0,1,2,3`, use `GPUS=0,1,2,3` in both jobs.

```bash
python codex_added/scripts/22_split_jsonl_for_jobs.py \
  --data data/private.jsonl \
  --out-dir codex_added/job_data \
  --prefix private \
  --num-jobs 2

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

See `docs/GETTING_STARTED.md` for the fuller workflow.
