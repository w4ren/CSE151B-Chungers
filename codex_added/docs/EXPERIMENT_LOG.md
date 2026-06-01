<!-- Added by Codex: experiment notes; not part of the original starter repository. -->

# Experiment Log

## Environment

- DataHub GPU: NVIDIA A30, 24GB VRAM.
- Torch: 2.2.1+cu121.
- Transformers was upgraded from 4.46.3 to 4.57.3 because 4.46.3 did not recognize `model_type: qwen3`.
- Transformers 5.6.2 was not usable with this DataHub Torch build because it requires PyTorch >= 2.4.

## Prompting Runs

### One-Problem Smoke

Command:

```bash
python codex_added/scripts/10_run_prompt_sweep.py \
  --input data/public.jsonl \
  --output codex_added/results/sweep_gpu_final_box_only_2048.jsonl \
  --limit 1 \
  --variants final_box_only \
  --num-samples 1 \
  --do-sample \
  --quantization none \
  --max-new-tokens 2048 \
  --batch-size 1 \
  --answer-key-mode strict
```

Result: `1/1` correct. The model emitted `\boxed{105950}` and stopped before the token limit.

### Five-Problem Slice

`concise_check`, 2048 tokens, one sample:

- Overall: `1/5`
- Free-form: `1/3`
- MCQ: `0/2`
- Runtime: about 5 minutes 18 seconds.

`final_box_only`, 2048 tokens, one sample:

- Overall: `2/5`
- Free-form: `2/3`
- MCQ: `0/2`
- Runtime: about 4 minutes 25 seconds.

Current takeaways:

- `final_box_only` is better than the initial concise prompt for answer formatting.
- MCQs are still too verbose and frequently hit the token limit before a clean answer.
- The tokenizer's default chat template forces `<think>`, so direct-mode prompting alone did not stop verbose reasoning.
- Prefixing `\boxed{` directly is not useful; the model placed prose inside the box.

## Next Experiments

1. Run self-consistency on a small slice with `final_box_only`, then compare strict vs loose answer-key voting.
2. Build and inspect answer-format SFT data with `12_build_answer_format_sft.py`.
3. Install optional training dependencies only after the prompting pipeline is stable.

## Answer-Format LoRA

Installed `peft==0.15.2` and trained a LoRA adapter on public answer-format examples:

```bash
python codex_added/scripts/13_train_lora_sft.py \
  --train codex_added/data/sft_answer_format.jsonl \
  --output-dir codex_added/models/qwen3_answer_format_lora \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 1536
```

Training details:

- 1112 examples used.
- 14 examples skipped for length.
- 33,030,144 trainable parameters, about 0.81% of the model.
- Runtime: about 6 minutes on the A30.
- Final average train loss: about 0.61.

Using the LoRA adapter directly makes formatting excellent but reasoning too shallow:

- First 5 public rows, direct LoRA, 512 token cap: `2/5`.
- Outputs were very short and boxed, but several answers were wrong.

The stronger use is two-stage:

1. Base Qwen generates reasoning.
2. The LoRA adapter finalizes the answer from the base reasoning trace.

Results on the first 20 public rows:

| Method | Overall | Free-form | MCQ |
|---|---:|---:|---:|
| Base reasoning, 1024 cap, raw strict extraction | 4/20 | 3/11 | 1/9 |
| Base reasoning, 1024 cap, LoRA finalizer | 8/20 | 6/11 | 2/9 |
| Base reasoning, 2048 cap, raw strict extraction | 7/20 | 6/11 | 1/9 |
| Base reasoning, 2048 cap, LoRA finalizer | 12/20 | 6/11 | 6/9 |

Best current pipeline:

```bash
python codex_added/scripts/10_run_prompt_sweep.py \
  --input data/public.jsonl \
  --output codex_added/results/sweep_public_20_base_reason_2048.jsonl \
  --limit 20 \
  --variants final_box_only \
  --num-samples 1 \
  --do-sample \
  --quantization none \
  --max-new-tokens 2048 \
  --batch-size 1 \
  --answer-key-mode strict

python codex_added/scripts/16_finalize_with_qwen.py \
  --data data/public.jsonl \
  --responses codex_added/results/sweep_public_20_base_reason_2048.jsonl \
  --output codex_added/results/finalized_public_20_base_reason_lora_full_2048.jsonl \
  --adapter-dir codex_added/models/qwen3_answer_format_lora \
  --max-new-tokens 64 \
  --temperature 0.1 \
  --score
```

Remaining failure buckets on that 20-row sample:

- 4 wrong free-form values.
- 3 wrong MCQ letters.
- 1 multi-answer count mismatch.

## Candidate Reranking And Selection

Self-consistency on the first 10 rows with `final_box_only`, 3 sampled 1024-token traces, and the LoRA finalizer:

- Overall: `6/10`.
- This beat one 1024-token trace (`5/10`) but did not beat one 2048-token trace (`7/10`).
- Most samples agreed exactly, so majority voting alone did not add enough diversity.

Added `17_rerank_with_qwen.py`, a same-model verifier/reranker. It gives Qwen multiple Qwen-generated candidate answers/traces and asks it to verify or solve independently. The reranker must be followed by the LoRA finalizer; raw reranker traces are often verbose or unfinished.

First 10 public rows:

| Method | Overall |
|---|---:|
| Base 2048 + LoRA finalizer | 7/10 |
| 3x1024 vote + LoRA finalizer | 6/10 |
| Mixed candidates, thinking rerank, LoRA finalizer | 8/10 |
| Mixed candidates, conflict-only thinking rerank, LoRA finalizer | 8/10 |
| Mixed candidates, direct conflict rerank, LoRA finalizer | 5/10 |

First 20 public rows using 1024 and 2048 candidates:

| Method | Overall | Free-form | MCQ |
|---|---:|---:|---:|
| Base 2048 + LoRA finalizer | 12/20 | 6/11 | 6/9 |
| Conflict-only thinking rerank + LoRA finalizer | 12/20 | 7/11 | 5/9 |
| Base 2048 for MCQ, reranked output for free-form | 13/20 | 7/11 | 6/9 |

Current best policy on the 20-row slice:

1. Use the base 2048-token `final_box_only` pipeline as the default.
2. Generate a second 1024-token candidate pass.
3. Rerank only candidate conflicts with `17_rerank_with_qwen.py --only-conflicts --assistant-mode think`.
4. Finalize reranked traces with the answer-format LoRA.
5. Use `18_select_predictions.py --policy free_form_override` so MCQs stay on the stronger 2048 baseline and free-form rows get the reranked output.

The `slot_exact` prompt variant was also tested with a 3072-token budget on the first 10 rows. It scored `6/10`, so it remains experimental and is not part of the default pipeline.

`19_run_best_pipeline.sh` now wraps the current best-known policy. With `LIMIT` set, it runs a public/dev slice and skips CSV creation. Without `LIMIT`, it writes a full competition CSV through `make_submission.py`.

## GRPO / RL Smoke Tests

Installed `trl==0.19.1` and added `14_train_grpo_public_reward.py`.

The script uses only Qwen/Qwen3-4B-Thinking-2507 plus a LoRA adapter. The reward function uses the public-set answer and the local competition judger:

- exact correct answer: default reward `1.0`;
- optional boxed/MCQ/slot-count shaping exists, but defaults to `0.0` because a first shaped run reinforced wrong boxed answers.

Important implementation finding:

- With gradient checkpointing enabled, TRL disabled generation cache and Qwen sampled clipped or unusable completions, producing zero reward.
- With `--no-gradient-checkpointing`, direct-mode GRPO produced short boxed completions and real reward variance.

Smoke runs:

| GRPO Run | Setup | Public Slice Result |
|---|---|---:|
| `qwen3_grpo_public_smoke` | thinking mode, 4 examples, 2 steps | reward stayed `0.0`; not useful |
| `qwen3_grpo_public_smoke_var` | direct mode, shaped reward, 8 examples, 5 steps | adapter direct scored `6/20`; as finalizer scored `12/20` |
| `qwen3_grpo_public_smoke_outcome` | direct mode, outcome-only reward, 8 examples, 5 steps | adapter direct scored `5/20` |
| `qwen3_grpo_public_outcome_120s_g2` | outcome-only reward, offset 100, 512 train examples, 120 steps, 2 generations, 768-token prompts | direct adapter scored `7/20`; as finalizer scored `12/20` |

Additional notes from the larger run:

- A 4-generation, 1024-token prompt run OOMed on the A30 after the first step. The working profile was `--num-generations 2 --max-prompt-length 768 --max-completion-length 256 --no-gradient-checkpointing`.
- The run trained on rows after offset 100, so the first-20 validation slice was held out.
- Reward remained sparse: many two-sample GRPO groups were both wrong or both right, giving zero advantage.
- Adding the GRPO direct output as a third candidate to the same-model reranker did not improve the selected result; finalized reranking scored `12/20`, and the best selection policy remains `13/20`.

Conclusion: GRPO is technically working and validated with a held-out slice, but the current larger run still does not beat the prompt/rerank pipeline. Do not use the GRPO adapter in the private submission path yet.

## External Data Added

Added a compact external SFT bundle under `codex_added/data/external/` for the next experiments. Exact normalized overlaps with `data/public.jsonl` were filtered out before use.

- `external_math_sft.jsonl`: 17,494 reasoning SFT examples.
- `external_math_finalizer_sft.jsonl`: 17,490 finalizer SFT examples built from the same worked traces.
- `external_math_manifest.json`: source counts and provenance.

Sources:

- `EleutherAI/hendrycks_math`, all train configs: 7,498 competition-style symbolic examples after filtering.
- `openai/gsm8k`, deterministic train sample: 2,500 arithmetic word-problem examples.
- `stellaathena/math_mcqa`, train split: 7,496 multiple-choice math examples after filtering.

Recommended next test is not to replace the current answer-format LoRA immediately. Train a separate adapter from `external_math_finalizer_sft.jsonl`, then compare it as the finalizer on the same first-20 public slice. If it does not regress, try a mixed finalizer dataset combining `codex_added/data/sft_answer_format.jsonl` with the external finalizer rows.
