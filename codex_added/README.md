<!-- Added by Codex: index for Codex-created files; not part of the original starter repository. -->

# Codex-Added Files

This folder collects the files added around the original starter repository.
The starter files at the repo root are unchanged.

## Layout

```text
codex_added/
  configs/        Baseline model and generation config
  data/           Derived training datasets
  docs/           Getting-started workflow notes
  math_comp/      Reusable data, prompt, inference, scoring, and submission code
  results/        Local generated JSONL outputs
  scripts/        CLI entry points
  submissions/    Local generated CSV submissions
  requirements.txt
```

## Common Commands

```bash
python codex_added/scripts/analyze_public.py --data data/public.jsonl
python codex_added/scripts/10_run_prompt_sweep.py --input data/public.jsonl --output codex_added/results/sweep_smoke.jsonl --limit 5 --variants final_box_only --num-samples 1 --do-sample --quantization none --max-new-tokens 2048 --batch-size 1
python codex_added/scripts/11_vote_self_consistency.py --data data/public.jsonl --responses codex_added/results/sweep_smoke.jsonl --output codex_added/results/voted_smoke.jsonl --score
python codex_added/scripts/12_build_answer_format_sft.py --public data/public.jsonl --output codex_added/data/sft_answer_format.jsonl
python codex_added/scripts/13_train_lora_sft.py --train codex_added/data/sft_answer_format.jsonl --output-dir codex_added/models/qwen3_answer_format_lora --max-examples 32 --epochs 1
python codex_added/scripts/14_train_grpo_public_reward.py --public data/public.jsonl --output-dir codex_added/models/qwen3_grpo_public_smoke --init-adapter-dir codex_added/models/qwen3_answer_format_lora --max-examples 8 --max-steps 5 --assistant-mode direct --num-generations 4 --logging-steps 10 --no-gradient-checkpointing
python codex_added/scripts/16_finalize_with_qwen.py --data data/public.jsonl --responses codex_added/results/sweep_smoke.jsonl --output codex_added/results/finalized_smoke.jsonl --adapter-dir codex_added/models/qwen3_answer_format_lora --score
python codex_added/scripts/17_rerank_with_qwen.py --data data/public.jsonl --responses codex_added/results/finalized_a.jsonl codex_added/results/finalized_b.jsonl --output codex_added/results/reranked.jsonl --only-conflicts
python codex_added/scripts/18_select_predictions.py --data data/public.jsonl --base codex_added/results/finalized_a.jsonl --override codex_added/results/reranked.jsonl --output codex_added/results/selected.jsonl --policy free_form_override --score
LIMIT=20 SCORE=1 codex_added/scripts/19_run_best_pipeline.sh data/public.jsonl codex_added/results/best_public_20 codex_added/submissions/best_public_20.csv
```

See `docs/GETTING_STARTED.md` for the fuller workflow.
