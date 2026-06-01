# Best Method

This is the current submission path. On the first 20 public rows it scored `13/20` overall, with `7/11` free-form and `6/9` MCQ correct.

## How It Works

The method uses Qwen twice to create two independent candidate traces:

1. A long `2048`-token `final_box_only` reasoning pass.
2. A shorter `1024`-token `final_box_only` reasoning pass.

Both traces are passed through the answer-format LoRA at `codex_added/models/qwen3_answer_format_lora`, which turns verbose Qwen reasoning into a clean boxed answer. When the two candidates disagree, base Qwen reranks only those conflicts in thinking mode. The final selection policy keeps the stronger 2048 baseline for MCQ rows and uses the reranked answer for free-form rows.

That last selection step is what improves the public slice from `12/20` to `13/20`.

## Run

Single process/full dataset:

```bash
codex_added/best/run.sh \
  data/private.jsonl \
  codex_added/results/best_private \
  codex_added/submissions/best_submission.csv
```

Visible multi-GPU machine:

```bash
GPUS=0,1,2,3,4,5,6,7 NUM_SHARDS=8 \
  codex_added/best/run_8gpu.sh \
  data/private.jsonl \
  codex_added/results/best_private_8gpu \
  codex_added/submissions/best_submission.csv
```

Two Kubernetes jobs with four GPUs each:

```bash
kubectl apply -f codex_added/best/k8s/cse151b-best-2x4gpu.yaml
```

The Kubernetes manifest first splits `data/private.jsonl` into `codex_added/job_data/private_job0.jsonl` and `private_job1.jsonl`, then runs the same 4-GPU shard launcher on each chunk. The merge job combines both selected JSONLs against the original full private file before writing the final CSV.

## Folder Contents

- `run.sh`: full best pipeline.
- `run_8gpu.sh`: best pipeline split across visible GPUs.
- `k8s/cse151b-best-2x4gpu.yaml`: prepare + job0 + job1 + merge Kubernetes workflow.
- `public_eval/`: first-20 public artifacts proving the `13/20` score.
- `adapter`: symlink to the active answer-format LoRA.
- `baseline.yaml`: symlink to the shared Qwen config.
