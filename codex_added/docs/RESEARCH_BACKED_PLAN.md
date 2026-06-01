# Research-Backed Plan, Edited for This Repo

This project should prioritize the stack that the current experiments and
hardware support:

1. Run larger Qwen-only reasoning budgets and measure them directly.
2. Make final answers slot-aware, boxed, and easy for the judger to parse.
3. Select among budget candidates by answer-key consensus and format quality.
4. Use SFT/RL only after candidate oracle accuracy and formatting failures are
   measured.

## Hardware Reality

The local pod has two RTX 2080 Ti GPUs. Plain fp16 vLLM with
`Qwen/Qwen3-4B-Thinking-2507` did not fit a 16k context: vLLM estimated only
about 3312 maximum sequence length after model load. The same 16k-capable run
does fit with vLLM bitsandbytes quantization.

For this pod, long-budget sweeps should use:

```bash
VLLM_QUANTIZATION=bitsandbytes
VLLM_LOAD_FORMAT=bitsandbytes
VLLM_GPU_MEMORY_UTILIZATION=0.95
VLLM_ENFORCE_EAGER=1
```

The active 30-problem sweep uses:

```bash
LIMIT=30 BUDGETS="4096 8192 16384" GPUS=0,1 \
  codex_added/scripts/24_run_vllm_token_sweeps_2gpu.sh \
  data/public.jsonl \
  codex_added/results/vllm_30_public_4k_8k_16k
```

## Immediate Evaluation Targets

After the 4k/8k/16k sweep finishes, inspect:

- Per-budget accuracy.
- Candidate oracle accuracy.
- Answer-key disagreement rate.
- `hit_token_limit` rate.
- `format_ok`, `expected_slots`, and `parsed_slots`.
- Whether consensus selection beats any single budget.

The sweep launcher now writes:

- `candidate_summary.json`
- `candidate_diagnostics.jsonl`
- `selected_consensus.jsonl`

## Inference Priority

Do not treat 2048 tokens as the main reasoning budget. Use it only as a cheap
screening budget. For this hardware, the realistic long-budget path is
quantized vLLM at 4k/8k/16k first, then longer only if the 16k run shows clear
oracle lift and acceptable runtime.

## Finalization Priority

The finalizer and reranker now condition on the expected answer schema:

- MCQ: exactly one boxed option letter.
- Free-form: exactly one boxed answer with exactly the expected number of
  ordered comma-separated slots.

Both scripts normalize recoverable outputs into a single boxed final answer and
record slot diagnostics.

## Selection Priority

Use budget-candidate consensus before doing more RL:

```bash
.venv-vllm/bin/python codex_added/scripts/25_analyze_candidate_pool.py \
  --data data/public.jsonl \
  --responses \
    codex_added/results/vllm_30_public_4k_8k_16k/4096/merged.jsonl \
    codex_added/results/vllm_30_public_4k_8k_16k/8192/merged.jsonl \
    codex_added/results/vllm_30_public_4k_8k_16k/16384/merged.jsonl \
  --labels 4096 8192 16384

.venv-vllm/bin/python codex_added/scripts/26_select_budget_candidates.py \
  --data data/public.jsonl \
  --responses \
    codex_added/results/vllm_30_public_4k_8k_16k/4096/merged.jsonl \
    codex_added/results/vllm_30_public_4k_8k_16k/8192/merged.jsonl \
    codex_added/results/vllm_30_public_4k_8k_16k/16384/merged.jsonl \
  --labels 4096 8192 16384 \
  --output codex_added/results/vllm_30_public_4k_8k_16k/selected_consensus.jsonl \
  --score
```

## Training Priority

Treat GRPO/RL as a later-stage step. The immediate higher-return work is:

1. Measure whether long-budget candidates contain correct answers.
2. Improve selection/finalization around those candidates.
3. Build verified self-generated traces and hard negatives from Qwen outputs.
4. Only then try RL or preference optimization with rewards for exact
   correctness, boxed format, slot completeness, and moderate length.
