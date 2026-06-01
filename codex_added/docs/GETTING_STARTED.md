<!-- Added by Codex: project workflow notes; not part of the original starter repository. -->

# Getting Started

This repo is the official starter code plus a small CLI workflow for faster iteration.
The original files are left intact.

## Setup

```bash
cd /home/w4ren/151B_SP26_Competition
python -m venv .venv
source .venv/bin/activate
pip install -r codex_added/requirements.txt
```

The competition requires final inference with `Qwen/Qwen3-4B-Thinking-2507`.
The default config uses the Transformers backend with 4-bit loading:

```bash
python codex_added/scripts/run_inference.py --config codex_added/configs/baseline.yaml --limit 5
```

On DataHub with an A30 GPU, start with unquantized Transformers if
`bitsandbytes` is not installed:

```bash
python codex_added/scripts/10_run_prompt_sweep.py \
  --input data/public.jsonl \
  --output codex_added/results/sweep_smoke.jsonl \
  --limit 5 \
  --variants starter_deep,answer_audit \
  --num-samples 2 \
  --do-sample \
  --quantization none \
  --max-new-tokens 512 \
  --batch-size 1
```

Vote and score the smoke traces:

```bash
python codex_added/scripts/11_vote_self_consistency.py \
  --data data/public.jsonl \
  --responses codex_added/results/sweep_smoke.jsonl \
  --output codex_added/results/voted_smoke.jsonl \
  --score
```

For faster answer-format experiments, remove the tokenizer's forced `<think>`
prefix and try the answer-first prompt:

```bash
python codex_added/scripts/10_run_prompt_sweep.py \
  --input data/public.jsonl \
  --output codex_added/results/sweep_direct_smoke.jsonl \
  --limit 5 \
  --variants boxed_first \
  --num-samples 1 \
  --do-sample \
  --assistant-mode direct \
  --quantization none \
  --max-new-tokens 512 \
  --batch-size 1
```

Use `--backend vllm` or `--quantization 4bit` only after those packages are
installed and verified.

## Inspect The Public Set

```bash
python codex_added/scripts/analyze_public.py --data data/public.jsonl
```

Current cloned public set has 1126 rows: 375 multiple-choice and 751 free-form.

## Score A Public Run

Run a smoke test:

```bash
python codex_added/scripts/run_inference.py \
  --config codex_added/configs/baseline.yaml \
  --data data/public.jsonl \
  --output codex_added/results/public_smoke.jsonl \
  --limit 5
```

Score an existing response JSONL:

```bash
python codex_added/scripts/evaluate_responses.py \
  --data data/public.jsonl \
  --predictions codex_added/results/public_smoke.jsonl \
  --output codex_added/results/public_smoke_scored.jsonl
```

For a voted prompt-sweep run, inspect failure buckets:

```bash
python codex_added/scripts/15_error_taxonomy.py \
  --data data/public.jsonl \
  --results codex_added/results/voted_smoke.jsonl
```

Build answer-format SFT data:

```bash
python codex_added/scripts/12_build_answer_format_sft.py \
  --public data/public.jsonl \
  --output codex_added/data/sft_answer_format.jsonl
```

Train a small answer-format LoRA adapter:

```bash
python codex_added/scripts/13_train_lora_sft.py \
  --train codex_added/data/sft_answer_format.jsonl \
  --output-dir codex_added/models/qwen3_answer_format_lora \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 1536
```

Run a small GRPO smoke test from the answer-format adapter:

```bash
python codex_added/scripts/14_train_grpo_public_reward.py \
  --public data/public.jsonl \
  --output-dir codex_added/models/qwen3_grpo_public_smoke \
  --init-adapter-dir codex_added/models/qwen3_answer_format_lora \
  --max-examples 8 \
  --max-steps 5 \
  --assistant-mode direct \
  --num-generations 4 \
  --max-completion-length 128 \
  --logging-steps 10 \
  --no-gradient-checkpointing
```

For Qwen3 on this A30, `--no-gradient-checkpointing` mattered: with checkpointing on, TRL disabled generation cache and the sampled completions failed to reach rewardable boxed answers.

Run inference with a trained adapter:

```bash
python codex_added/scripts/10_run_prompt_sweep.py \
  --input data/public.jsonl \
  --output codex_added/results/sweep_public_lora_smoke.jsonl \
  --limit 5 \
  --variants final_box_only \
  --num-samples 1 \
  --do-sample \
  --quantization none \
  --adapter-dir codex_added/models/qwen3_answer_format_lora \
  --max-new-tokens 1024 \
  --batch-size 1 \
  --answer-key-mode strict
```

Use the same Qwen model as a second-stage final-answer formatter:

```bash
python codex_added/scripts/16_finalize_with_qwen.py \
  --data data/public.jsonl \
  --responses codex_added/results/sweep_public_lora_smoke.jsonl \
  --output codex_added/results/finalized_public_lora_smoke.jsonl \
  --adapter-dir codex_added/models/qwen3_answer_format_lora \
  --only-missing \
  --score
```

## Current Best Public-Slice Pipeline

The strongest tested policy on the first 20 public rows is:

1. Generate a 2048-token base Qwen pass.
2. Generate a 1024-token base Qwen pass.
3. Finalize both with the answer-format LoRA.
4. Rerank only answer conflicts with base Qwen in thinking mode.
5. Finalize the reranker output with the LoRA.
6. Keep the 2048 baseline for MCQ rows and use the reranked output for free-form rows.

Selection command after those intermediate files exist:

```bash
python codex_added/scripts/18_select_predictions.py \
  --data data/public.jsonl \
  --base codex_added/results/finalized_public_20_base_reason_lora_full_2048.jsonl \
  --override codex_added/results/finalized_public_20_reranked_1024_2048_think_conflicts.jsonl \
  --output codex_added/results/selected_public_20_base2048_rerank_freeform.jsonl \
  --policy free_form_override \
  --limit 20 \
  --score
```

This scored `13/20` on that slice, versus `12/20` for the 2048-token baseline alone.

## Build A Private Submission

Place the competition `private.jsonl` at:

```text
data/private.jsonl
```

Then run:

```bash
codex_added/scripts/19_run_best_pipeline.sh \
  data/private.jsonl \
  codex_added/results/best_private \
  codex_added/submissions/best_submission.csv
```

The CSV writer preserves raw model responses and uses the private file for id
ordering, which matches the required `id,response` submission format.
