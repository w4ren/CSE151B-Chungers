# LoRA Improvement Workflow

This note describes additive scripts for controlled LoRA experiments. They do not replace the current best pipeline unless a locked public holdout shows a win.

## Rule

Use private rows for inference and schema compatibility only. Do not train on private labels or guessed private answers.

Use public labels only through a locked split:

- Train IDs: allowed for SFT examples, synthetic variants, and checkpoint selection inputs.
- Holdout IDs: never used for training, synthetic variants, prompt tuning, rule design, or checkpoint selection.

## Scripts

- `60_build_mcq_selector_sft.py`: builds MCQ selector SFT data from raw traces, candidate outputs, and optional baseline predictions.
- `61_build_finalizer_schema_sft.py`: builds hard-negative finalizer/schema SFT data.
- `62_build_solver_rejection_sft.py`: builds solver SFT data from correct self-generated traces.
- `63_train_chat_lora_sft.py`: generic chat-message LoRA/QLoRA SFT trainer for all three datasets.
- `64_eval_category_lora.py`: runs an MCQ selector or free-form finalizer adapter on public raw traces.
- `65_route_category_lora_outputs.py`: combines category-specific LoRA outputs with a baseline prediction file.

Existing scripts to reuse:

- Raw Qwen generation: `10_run_prompt_sweep.py`
- Current finalizer: `16_finalize_with_qwen.py`
- Scoring: `evaluate_responses.py`, `32_score_stage_progression.py`
- MCQ best-of/candidate logic: `42_select_mcq_bestof.py`, `50_run_mcq_trace_extract_logprob.py`
- Deterministic free-form/schema passes: `39_precision_form_normalize.py`, `40_deterministic_numeric_finalizer.py`, `44_slot_aware_finalize.py`, `52_schema_aware_freeform_finalize.py`

## 1. MCQ Selector LoRA

Purpose: prevent MCQ finalizer flips and choose one option letter from raw traces/candidates.

Build data:

```bash
python codex_added/scripts/60_build_mcq_selector_sft.py \
  --public data/public.jsonl \
  --raw-responses codex_added/results/public_full_raw8k_20260531_090737/raw_8k_merged.jsonl \
  --candidate-responses codex_added/results/public100_rebuilt_20260531_084047/v7_derivative_mcq_canonicalized.jsonl \
  --baseline codex_added/results/public100_rebuilt_20260531_084047/v7_derivative_mcq_canonicalized.jsonl \
  --output-dir codex_added/data/mcq_selector_lora \
  --permutations-per-example 1
```

Train:

```bash
python codex_added/scripts/63_train_chat_lora_sft.py \
  --train codex_added/data/mcq_selector_lora/train.jsonl \
  --eval codex_added/data/mcq_selector_lora/holdout.jsonl \
  --output-dir codex_added/models/qwen3_mcq_selector_lora \
  --max-length 2048 \
  --learning-rate 2e-5 \
  --lora-r 16 \
  --lora-alpha 32 \
  --epochs 1 \
  --save-steps 25
```

Evaluate:

```bash
python codex_added/scripts/64_eval_category_lora.py \
  --mode mcq_selector \
  --data data/public.jsonl \
  --raw-responses codex_added/results/public_full_raw8k_20260531_090737/raw_8k_merged.jsonl \
  --adapter-dir codex_added/models/qwen3_mcq_selector_lora \
  --output codex_added/results/mcq_selector_eval/preds.jsonl \
  --score \
  --backend vllm \
  --max-new-tokens 16 \
  --vllm-batch-size 16
```

Use only for MCQ rows if it improves locked MCQ accuracy without large flip losses.

## 2. Finalizer/Schema LoRA

Purpose: convert solver traces into clean final answers, especially multi-slot free-form rows.

Build data:

```bash
python codex_added/scripts/61_build_finalizer_schema_sft.py \
  --public data/public.jsonl \
  --raw-responses codex_added/results/public_full_raw8k_20260531_090737/raw_8k_merged.jsonl \
  --current-finalizer codex_added/results/public100_rebuilt_20260531_084047/lora_finalized.jsonl \
  --output-dir codex_added/data/finalizer_schema_lora \
  --synthetic-corruptions-per-row 2 \
  --oversample-multislot 3
```

Train, usually starting from the supervised finalizer:

```bash
python codex_added/scripts/63_train_chat_lora_sft.py \
  --train codex_added/data/finalizer_schema_lora/train.jsonl \
  --eval codex_added/data/finalizer_schema_lora/holdout.jsonl \
  --output-dir codex_added/models/qwen3_finalizer_schema_lora \
  --init-adapter-dir codex_added/models/qwen3_answer_format_lora_rebuilt_20260531_070427 \
  --max-length 2048 \
  --learning-rate 1e-5 \
  --lora-r 16 \
  --lora-alpha 32 \
  --epochs 1 \
  --save-steps 25
```

Evaluate:

```bash
python codex_added/scripts/64_eval_category_lora.py \
  --mode finalizer_schema \
  --data data/public.jsonl \
  --raw-responses codex_added/results/public_full_raw8k_20260531_090737/raw_8k_merged.jsonl \
  --current-finalizer codex_added/results/public100_rebuilt_20260531_084047/lora_finalized.jsonl \
  --adapter-dir codex_added/models/qwen3_finalizer_schema_lora \
  --output codex_added/results/finalizer_schema_eval/preds.jsonl \
  --score \
  --backend vllm \
  --max-new-tokens 128 \
  --vllm-batch-size 8
```

Use only for free-form/schema-sensitive rows if it improves locked free-form or multi-slot accuracy.

## 3. Solver Rejection-Sampling LoRA

Purpose: improve reasoning by imitating correct self-generated traces.

Build data from candidate traces:

```bash
python codex_added/scripts/62_build_solver_rejection_sft.py \
  --public data/public.jsonl \
  --responses codex_added/results/public_full_raw8k_20260531_090737/raw_8k_merged.jsonl \
  --output-dir codex_added/data/solver_rejection_lora \
  --max-traces-per-id 3 \
  --exclude-token-limit
```

Train:

```bash
python codex_added/scripts/63_train_chat_lora_sft.py \
  --train codex_added/data/solver_rejection_lora/train.jsonl \
  --eval codex_added/data/solver_rejection_lora/holdout.jsonl \
  --output-dir codex_added/models/qwen3_solver_rejection_lora \
  --max-length 4096 \
  --learning-rate 1e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --epochs 1 \
  --save-steps 50 \
  --use-4bit
```

Evaluate the solver only through the full finalizer stack. Do not judge it by training loss.

## 4. Category Routing

After evaluating category adapters, combine them over a baseline:

```bash
python codex_added/scripts/65_route_category_lora_outputs.py \
  --data data/public.jsonl \
  --base codex_added/results/public100_rebuilt_20260531_084047/v7_derivative_mcq_canonicalized.jsonl \
  --mcq-selector codex_added/results/mcq_selector_eval/preds.jsonl \
  --schema-finalizer codex_added/results/finalizer_schema_eval/preds.jsonl \
  --output codex_added/results/category_routed_lora_eval/preds.jsonl \
  --score
```

The private route should use the same learned routing policy selected on locked public holdout. Do not tune routing from private correctness.

## Hyperparameters

MCQ selector:

- rank: `8-16`
- alpha: `2 * rank`
- dropout: `0.0-0.05`
- LR: `1e-5` to `3e-5`
- max length: `1024-2048`
- epochs: `1-3`

Finalizer/schema:

- rank: `8-16`
- alpha: `2 * rank`
- dropout: `0.0-0.05`
- LR: `1e-5` to `5e-5`
- max length: `1024-2048`
- epochs: `1-2`

Solver rejection SFT:

- rank: `16-64`
- alpha: `2 * rank`
- dropout: `0.05`
- LR: `5e-6` to `2e-5`
- max length: `4096-8192`
- train by checkpoints, not final epoch

## Decision Rule

Do not replace the current pipeline globally unless locked public holdout improves.

Route by category:

- MCQ rows: MCQ selector LoRA only if MCQ holdout improves.
- Free-form rows: schema/finalizer LoRA only if free-form or multi-slot holdout improves.
- Hard reasoning rows: solver LoRA only if full pipeline improves, not just raw trace quality.
- Formatting-sensitive rows: renderer/finalizer LoRA only if it does not introduce losses.

If a new adapter ties the supervised finalizer, keep the supervised finalizer.

## Runtime Estimates

Approximate runtimes:

| Experiment | 1x H100/H200 | 4x RTX PRO 6000 |
|---|---:|---:|
| MCQ selector build | minutes | minutes |
| MCQ selector train | 20-60 min | 30-90 min |
| Finalizer/schema build | minutes | minutes |
| Finalizer/schema train | 30-90 min | 45-120 min |
| Solver rejection data after traces | minutes | minutes |
| Solver rejection train | 6-18h | 10-24h |
| Category eval | 15-60 min | 15-60 min |

Inference over private with the current raw 8k pipeline is about 1.5 hours on 4 GPUs for one sample, plus a few minutes for finalizer/postprocess. Best-of-N scales with number of samples unless parallelized across more GPUs.

## Recommended Order

1. MCQ selector LoRA.
2. Finalizer/schema LoRA with hard negatives.
3. Rejection-sampling solver LoRA.
4. GRPO only after the above SFT baselines are solid.
