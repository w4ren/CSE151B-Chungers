# Second-Best Method

This is the strongest simple baseline. On the first 20 public rows it scored `12/20` overall, with `6/11` free-form and `6/9` MCQ correct.

## How It Works

The method runs one long `2048`-token Qwen reasoning pass with the `final_box_only` prompt. It then uses the answer-format LoRA as a second-stage finalizer. The LoRA does not solve the problem from scratch; it reads the problem plus Qwen's previous trace and emits only the final boxed answer.

This method is simpler and faster than the best pipeline because it does not run a second 1024-token pass, rerank conflicts, or apply a final selection policy. It ties the third method overall but has better MCQ accuracy, so it is ranked second.

## Run

```bash
codex_added/second_best/run.sh \
  data/private.jsonl \
  codex_added/results/second_best_private \
  codex_added/submissions/second_best_submission.csv
```

For split data, pass a chunk JSONL as the first argument. Example after running `22_split_jsonl_for_jobs.py`:

```bash
codex_added/second_best/run.sh \
  codex_added/job_data/private_job0.jsonl \
  codex_added/results/second_best_job0 \
  codex_added/submissions/second_best_job0_unused.csv
```

Merge multiple chunk outputs with `codex_added/scripts/21_merge_shard_predictions.py` against the original full dataset.

## Folder Contents

- `run.sh`: 2048-pass + LoRA finalizer pipeline.
- `public_eval/`: first-20 public artifacts proving the `12/20` score.
- `adapter`: symlink to the active answer-format LoRA.
- `baseline.yaml`: symlink to the shared Qwen config.
