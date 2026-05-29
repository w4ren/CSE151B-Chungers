# Raw Qwen Grid Search Handoff

This folder is the lightweight handoff for running raw-Qwen hyperparameter sweeps and comparing against the current best dev pipeline.

## Current Reference Scores

Public200 dev/audit slice:

| Stage | Overall | MCQ | Free-form |
|---|---:|---:|---:|
| Raw Qwen 8k | `105/200` | `47/78` | `58/122` |
| LoRA finalizer | `130/200` | `57/78` | `73/122` |
| V4 precision/form normalizer | `137/200` | `57/78` | `80/122` |
| V5 deterministic numeric finalizer | `152/200` | `57/78` | `95/122` |
| V6 slot-aware finalizer | `158/200` | `57/78` | `101/122` |

Use `158/200` only as a leaky dev metric. The deterministic finalizers were built after inspecting this public200 slice.

## Most Important Files

Raw generation / grid search:

- `codex_added/scripts/10_run_prompt_sweep.py`: main raw-Qwen generation entrypoint. Important knobs: `--variants`, `--max-new-tokens`, `--temperature`, `--top-p`, `--top-k`, `--num-samples`, `--assistant-mode`, `--backend vllm`.
- `codex_added/scripts/24_run_vllm_token_sweeps_2gpu.sh`: two-GPU token-budget sweep wrapper. Good starting point for raw-Qwen grid search.
- `codex_added/math_comp/variants.py`: prompt variants. Start with `final_box_only`, `slot_exact`, `exact_conservative`, `symbolic_strict`, and maybe `mcq_concise_compute`.
- `codex_added/math_comp/prompts.py`: chat prompt construction and default systems.
- `codex_added/math_comp/vllm_helpers.py`: vLLM loading/sampling helpers.
- `codex_added/math_comp/scoring.py` and `judger.py`: extraction and scoring. Always score with the updated judger.

Postprocess / score:

- `codex_added/scripts/21_merge_shard_predictions.py`: merge GPU shard outputs in data order.
- `codex_added/scripts/25_analyze_candidate_pool.py`: inspect candidate correctness and oracle accuracy when gold exists.
- `codex_added/scripts/26_select_budget_candidates.py`: select among token-budget candidates.
- `codex_added/scripts/32_score_stage_progression.py`: score multiple stages side by side and write per-stage scored JSONL.
- `codex_added/scripts/39_precision_form_normalize.py`: V4 deterministic precision/form normalizer.
- `codex_added/scripts/40_deterministic_numeric_finalizer.py`: V5 deterministic numeric recompute pass.
- `codex_added/scripts/44_slot_aware_finalize.py`: V6 guarded slot-aware finalizer.

Known bad MCQ experiment:

- `codex_added/scripts/45_run_public200_mcq_direct_bestofN_1gpu.sh`
- `codex_added/scripts/47_run_public200_mcq_direct_bestofN_2gpu.sh`

These are included for reproducibility, but the run regressed MCQ from `57/78` to `22/78`; do not use that path as the next default.

## Solved Problem Files

- `public200_stage_solved_summary.json`: solved/wrong IDs by stage, including raw-to-V6 gains/losses.
- `public200_solved_ids.md`: compact human-readable solved/wrong ID summary.
- `public200_v6_solved_rows.jsonl`: one row per V6-solved problem with ID, type, gold, V6 answer key, raw answer key, and a question preview.

The most useful raw-Qwen tuning split is:

- Regression controls: V6-solved IDs, especially raw-solved IDs.
- Opportunity set: raw-wrong but V6-correct IDs.
- Hard remaining set: V6-wrong IDs.

Do not tune only on the V6-wrong set; that will overfit and may break many already-correct rows.

## Suggested Grid

Start small before launching a full run:

```bash
cd /home/mnt/BiomechAI/Wenhao/archive2/CSE151B-Chungers-wenhao-lfs

PYTHON=.venv-vllm/bin/python \
LIMIT=30 \
BUDGETS="4096 8192 12288" \
GPUS=0,1 \
./codex_added/scripts/24_run_vllm_token_sweeps_2gpu.sh \
  data/public.jsonl \
  codex_added/results/raw_qwen_grid_smoke
```

For full public200-style evaluation, use the same script with a curated slice and compare raw-stage score plus downstream V4/V5/V6 score. The target is not just higher raw accuracy; it must improve or preserve final pipeline accuracy after deterministic postprocessing.

Recommended first grid:

| Knob | Values |
|---|---|
| Prompt variant | `final_box_only`, `slot_exact`, `exact_conservative`, `symbolic_strict` |
| Max new tokens | `4096`, `8192`, `12288`, `16384` |
| Temperature | `0.2`, `0.4`, `0.6` |
| Top-p | `0.85`, `0.95` |
| Top-k | `20`, `40` |
| Samples | `1` first, then `2-4` only for promising configs |

Use MCQ regression controls because prior direct-MCQ prompting collapsed badly. More tokens may help a few rows, but on V6 wrong MCQs `19/21` already hit the 8192 cap, so token increases alone are unlikely to be the main fix.
