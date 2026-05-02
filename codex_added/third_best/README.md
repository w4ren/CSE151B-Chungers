# Third-Best Method

This is the conflict-rerank method without the final best-method selection policy. On the first 20 public rows it scored `12/20` overall, with `7/11` free-form and `5/9` MCQ correct.

## How It Works

The method generates two finalized candidates for every problem:

1. A `2048`-token base Qwen reasoning pass finalized by the answer-format LoRA.
2. A `1024`-token base Qwen reasoning pass finalized by the answer-format LoRA.

If the answer keys agree, the reranker keeps the unanimous candidate. If they disagree, base Qwen sees both candidates and reranks/synthesizes a new trace in thinking mode. That reranked trace is then passed through the same answer-format LoRA to produce the final boxed response.

This improves free-form accuracy over the second method on the public slice, but it hurts MCQ accuracy. Because the final selection policy is omitted, it stays behind the best method.

## Run

```bash
codex_added/third_best/run.sh \
  data/private.jsonl \
  codex_added/results/third_best_private \
  codex_added/submissions/third_best_submission.csv
```

For split data, pass a chunk JSONL as the first argument. Example after running `22_split_jsonl_for_jobs.py`:

```bash
codex_added/third_best/run.sh \
  codex_added/job_data/private_job1.jsonl \
  codex_added/results/third_best_job1 \
  codex_added/submissions/third_best_job1_unused.csv
```

Merge multiple chunk outputs with `codex_added/scripts/21_merge_shard_predictions.py` against the original full dataset.

## Folder Contents

- `run.sh`: 2048 + 1024 candidate generation, conflict reranking, and LoRA finalization.
- `public_eval/`: first-20 public artifacts proving the `12/20` score.
- `adapter`: symlink to the active answer-format LoRA.
- `baseline.yaml`: symlink to the shared Qwen config.
