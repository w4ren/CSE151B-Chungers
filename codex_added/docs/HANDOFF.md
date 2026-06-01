<!-- Added by Codex: concise handoff notes for collaborators. -->

# Handoff

This repo is the original competition starter plus the `codex_added/` workflow.
For a collaborator, the useful files are the code and docs, not the local model
adapters or generated result JSONLs.

## Start Here

1. Read `README.md` at the repo root for the original starter context.
2. Read `codex_added/docs/GETTING_STARTED.md` for setup and commands.
3. Read `codex_added/docs/EXPERIMENT_LOG.md` for what has already been tried.
4. Use `codex_added/scripts/19_run_best_pipeline.sh` for the current best
   private-submission pipeline.

## Files Worth Sharing

- `codex_added/README.md`: index of added files and common commands.
- `codex_added/requirements.txt`: Python package pins used by the experiments.
- `codex_added/configs/baseline.yaml`: default Qwen config.
- `codex_added/math_comp/`: reusable data, prompt, inference, scoring, and
  submission helpers.
- `codex_added/scripts/`: CLI scripts for sweeps, voting, finalization,
  reranking, training, and submission creation.
- `codex_added/docs/GETTING_STARTED.md`: practical workflow notes.
- `codex_added/docs/EXPERIMENT_LOG.md`: score comparisons and conclusions.

## Current Best Known Method

On the first 20 public rows, the best tested policy scored `13/20`:

1. Generate a 2048-token base Qwen reasoning pass.
2. Generate a 1024-token base Qwen reasoning pass.
3. Finalize both with the answer-format LoRA.
4. Rerank only answer conflicts in Qwen thinking mode.
5. Finalize the reranked output.
6. Keep the 2048 baseline for MCQ rows and use reranked output for free-form
   rows.

The wrapper is:

```bash
codex_added/scripts/19_run_best_pipeline.sh \
  data/private.jsonl \
  codex_added/results/best_private \
  codex_added/submissions/best_submission.csv
```

## What Is Intentionally Not Shared

- `codex_added/models/`: local LoRA adapters and tokenizer artifacts.
- `codex_added/results/`: generated experiment outputs.
- `codex_added/submissions/*.csv`: local submission files.
- `codex_added/data/`: generated SFT/external data files.

Those can be rebuilt from the scripts and docs if needed. Keeping them out of
Git avoids a multi-gigabyte repository and accidental public-data exposure.
