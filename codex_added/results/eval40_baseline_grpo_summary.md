# Eval40 Baseline and GRPO Summary

Dataset:

- `codex_added/job_data/public_stratified_40.jsonl`
- 40 rows, stratified as 20 MCQ and 20 free-form.
- Held-out ids are listed in `codex_added/job_data/public_stratified_40_ids.txt`.

Completed results:

| Variant | Eval mode | MCQ | Free-form | Overall | Balanced | Public-weighted |
|---|---:|---:|---:|---:|---:|---:|
| SFT LoRA finalizer baseline / Method 1 | finalizer | 12/20 | 14/20 | 26/40 | 0.6500 | 0.6667 |
| GRPO direct | direct generation | 8/20 | 9/20 | 17/40 | 0.4250 | 0.4334 |
| GRPO finalizer | finalizer | 11/20 | 14/20 | 25/40 | 0.6250 | 0.6501 |

Artifacts:

- SFT LoRA finalizer baseline / Method 1: `codex_added/results/best_public_stratified_40_2gpu/finalized_2048.jsonl`
- GRPO direct eval: `codex_added/results/grpo_eval40_direct_512p128c_scored.jsonl`
- GRPO finalizer eval: `codex_added/results/grpo_eval40_finalizer_512p128c_scored.jsonl`
- GRPO adapter: `codex_added/models/qwen3_grpo_exclude_eval40_512p128c`

GRPO setup:

- Started from SFT LoRA adapter: `codex_added/models/qwen3_answer_format_lora`
- Excluded the 40 eval ids using `codex_added/job_data/public_stratified_40_ids.txt`
- Direct assistant mode
- Outcome-only reward
- `num_generations=2`
- `max_examples=256`
- `max_steps=60`
- `max_prompt_length=512`
- `max_completion_length=128`
- No gradient checkpointing
- The intended 768/256 profile OOMed on local 11 GB RTX 2080 Ti GPUs, so this was a reduced-token run.

Takeaway:

GRPO did not beat the supervised LoRA finalizer. Direct GRPO was much weaker, and GRPO-as-finalizer matched free-form but lost one MCQ compared with SFT.
