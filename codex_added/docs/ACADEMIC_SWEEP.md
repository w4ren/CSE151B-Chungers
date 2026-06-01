<!-- Added by Codex: literature-backed improvement plan for the current competition pipeline. -->

# Academic Sweep: Improving The Current Qwen Math Pipeline

Date: 2026-05-26

This note maps the current pipeline onto the academic literature on math LLMs,
verifiers, self-consistency, process supervision, and test-time compute. It is
written as an experiment plan, not as a survey for its own sake.

## Current Baseline To Improve

The current repo has three relevant methods:

- Raw Qwen reasoning with `Qwen/Qwen3-4B-Thinking-2507`.
- A LoRA finalizer that turns a Qwen reasoning trace into a clean boxed answer.
- A two-candidate conflict reranker using 1024-token and 2048-token candidates.

The strongest held-out result in `main.tex` is not the first-20 "best" rule. On
the 40-row held-out slice:

| Method | MCQ | Free-form | Overall | Public-weighted |
|---|---:|---:|---:|---:|
| Raw Qwen 2048 | 4/20 | 10/20 | 14/40 | 0.4001 |
| Qwen 2048 + LoRA finalizer | 11/20 | 14/20 | 25/40 | 0.6501 |
| 1024/2048 rerank + LoRA | 12/20 | 13/20 | 25/40 | 0.6334 |
| Problem-type selection | 11/20 | 13/20 | 24/40 | 0.6167 |

The first conclusion is that the LoRA finalizer is the most reliable current
gain. Reranking is promising, but the current selection policy overfit the first
20 rows and did not generalize.

## Literature Takeaways

### 1. More Samples Help Only If Selection Improves

Chain-of-thought prompting improves math reasoning by encouraging intermediate
reasoning, but a single sample is brittle:

- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language
  Models" - https://arxiv.org/abs/2201.11903
- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language
  Models" - https://arxiv.org/abs/2203.11171

Self-consistency works by sampling multiple reasoning paths and aggregating the
final answers. This repo tried a small self-consistency run and saw limited gain
because the samples were not diverse enough. The lesson is not "voting failed";
it is "same-prompt, same-model, few-sample voting is underpowered."

Universal Self-Consistency is closer to this repo's reranker idea because it
asks an LLM to choose among heterogeneous candidate outputs rather than requiring
identical normalized answers:

- Chen et al., "Universal Self-Consistency for Large Language Model Generation"
  - https://arxiv.org/abs/2311.17311

Practical implication: generate more meaningfully different candidates, then
select with a verifier or structured consistency procedure. Do not only compare
1024 vs 2048 traces from the same prompt.

### 2. Verifiers Scale Better Than Blind Generation

The classic GSM8K verifier result generated many solutions and trained a model
to rank them:

- Cobbe et al., "Training Verifiers to Solve Math Word Problems"
  - https://arxiv.org/abs/2110.14168

OpenAI's process-supervision work found that supervising intermediate reasoning
steps can outperform outcome-only supervision on MATH:

- Lightman et al., "Let's Verify Step by Step"
  - https://arxiv.org/abs/2305.20050

Qwen2.5-Math used a related self-improvement loop: sample, score with a reward
model, improve SFT data, retrain, and use reward-guided inference:

- Yang et al., "Qwen2.5-Math Technical Report"
  - https://arxiv.org/abs/2409.12122

Practical implication: the current generative reranker is too informal. A
pointwise or pairwise Qwen verifier trained on this repo's generated candidates
could select answers more consistently and avoid overwriting correct answers.

### 3. Rejection-Sampled Fine-Tuning Is A Better Next Step Than More Prompting

Rejection sampling fine-tuning (RFT) generates multiple solutions, keeps correct
reasoning paths according to a verifier/judge, then fine-tunes on those paths:

- Yuan et al., "Scaling Relationship on Learning Mathematical Reasoning with
  Large Language Models" - https://arxiv.org/abs/2308.01825

This is well matched to the competition because the public set has labels and a
local judge. It also fits the existing scripts: generate many Qwen samples,
score them with `judger.py`, keep correct traces, and train a solver or
finalizer LoRA.

Practical implication: after the answer-format LoRA, the next training target
should be either:

- a solver LoRA trained on correct Qwen traces; or
- a verifier/finalizer LoRA trained on candidate traces and the gold final
  answer.

### 4. GRPO Needs Denser Rewards Than The Current Run

DeepSeekMath introduced GRPO and showed that rule-based rewards can improve math
reasoning:

- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
  Open Language Models" - https://arxiv.org/abs/2402.03300

This repo's GRPO attempts technically worked but did not beat supervised
finalization. That is unsurprising: the reward was sparse, the slice was small,
and many generated groups had no within-group advantage signal.

Practical implication: do not make GRPO the next main path. If revisited, add
dense reward terms for extractability, slot count, MCQ letter validity, symbolic
equivalence, and partial credit on multi-slot rows.

### 5. Small Models Benefit From Search, But Search Needs A Scorer

Tree-of-Thoughts and rStar-Math show that deliberate search can improve
reasoning when the model can evaluate partial paths:

- Yao et al., "Tree of Thoughts" - https://arxiv.org/abs/2305.10601
- Guan et al., "rStar-Math" - https://arxiv.org/abs/2501.04519

rStar-Math is especially relevant because it targets small language models and
uses MCTS guided by a process reward model. A full MCTS system is probably too
heavy for this competition timeline, but a shallow version is practical:
generate several candidates, verify each, and only spend extra compute on hard
or conflicting rows.

Practical implication: implement adaptive test-time compute rather than a fixed
pipeline for every row.

### 6. Synthetic Math Data Helps, But Format Match Matters

Math-specific SFT datasets and instruction evolution help open models:

- MetaMath - https://arxiv.org/abs/2309.12284
- WizardMath - https://arxiv.org/abs/2308.09583
- MAmmoTH - https://arxiv.org/abs/2309.05653
- MathScale - https://arxiv.org/abs/2403.02884
- OpenMathInstruct-1 - https://arxiv.org/abs/2402.10176

The repo already added external SFT data. The risk is distribution mismatch:
competition rows require exact `\boxed{}` responses, ordered multi-slot answers,
and MCQ letters. External traces that solve math but do not match the judge's
format can make the finalizer worse.

Practical implication: external data should be transformed into the competition
format before training. Use it to teach reasoning, but always validate against a
held-out slice with the local judge.

### 7. MCQ Rows Need Separate Treatment

MCQ LLM outputs are sensitive to option order and option labels:

- Zheng et al., "Large Language Models Are Not Robust Multiple Choice Selectors"
  - https://arxiv.org/abs/2309.03882
- Pezeshkpour and Hruschka, "Large Language Models Sensitivity to the Order of
  Options in Multiple-Choice Questions" - https://arxiv.org/abs/2308.11483
- Wang et al., "Look at the Text: Instruction-Tuned Language Models are More
  Robust Multiple Choice Selectors than You Think" - https://arxiv.org/abs/2404.08382

The repo's MCQ performance improved dramatically with the finalizer on the first
20 rows, but held-out performance is still weak. Since MCQ rows have only one
letter answer, they are good candidates for option permutation and text-answer
matching.

Practical implication: for MCQ, generate the answer without relying on the
letter, map it to options, and/or majority vote across shuffled option orders.

## Highest-Value Improvements

### Priority 0: Fix Evaluation Before More Optimization

Problem: first-20 tuning produced a selection policy that did not generalize.

Actions:

1. Create a fixed validation split excluding ids 0-19.
2. Stratify by row type: MCQ, free-form 1 slot, free-form 2 slots, free-form 3+
   slots.
3. Report overall, MCQ, free-form, public-weighted accuracy, and bootstrap
   confidence intervals.
4. Freeze the split and stop choosing methods on the first 20 rows.

Why: without a reliable validation protocol, prompt and rerank changes will keep
overfitting tiny slices.

Expected effort: low.

Expected payoff: high, because it prevents false progress.

### Priority 1: Make The Finalizer Slot-Aware

Problem: multi-answer rows can collapse to one answer, especially when the trace
contains many intermediate quantities.

Actions:

1. Add `slot_count` and explicit slot instructions to finalizer prompts.
2. Train a separate finalizer dataset that oversamples multi-slot public rows.
3. Add synthetic multi-slot examples from external data transformed to the
   competition format.
4. Add a post-finalizer check: if a free-form problem has `k` slots and the
   final answer has not exactly `k` comma-separated fields, retry finalization
   with a stricter slot prompt.

Why: current biggest residual free-form failure is incomplete final answers, not
necessarily absent reasoning.

Expected effort: low to medium.

Expected payoff: high on free-form rows, which dominate the public distribution.

### Priority 2: Replace The Generative Reranker With A Verifier Score

Problem: the current reranker sometimes overwrites a correct answer.

Actions:

1. Build candidate tables from existing generated files:
   `id`, `problem`, `candidate_response`, `answer_key`, `source_trace`,
   `is_correct` on public data.
2. Train a Qwen LoRA verifier to answer a constrained question:
   "Is this candidate final answer correct for this problem? output
   \boxed{yes} or \boxed{no}."
3. At inference, score each candidate independently and pick the highest
   confidence candidate.
4. Keep the old 2048 + finalizer answer if verifier confidence is weak or tied.

Why: verifier literature consistently says selection is a separate task from
generation. The current same-model reranker mixes verification, solving, and
formatting in one generation.

Expected effort: medium.

Expected payoff: high if enough public/external labeled candidate data is
generated.

### Priority 3: Rejection-Sampled Solver LoRA

Problem: the finalizer fixes format but not wrong reasoning.

Actions:

1. Generate `N=8-32` candidates per public problem with varied prompts,
   temperature, and token budgets.
2. Score with the local judge.
3. Keep correct traces as SFT data.
4. Train a solver LoRA on correct traces, separate from the finalizer LoRA.
5. Evaluate:
   - base Qwen 2048 + finalizer;
   - solver LoRA 2048 + finalizer;
   - solver LoRA candidates + verifier.

Why: RFT is a direct fit for public labels and avoids the sparse-reward problem
that hurt GRPO.

Expected effort: medium to high.

Expected payoff: medium to high, especially if candidate diversity increases.

### Priority 4: MCQ Option-Permutation Ensemble

Problem: MCQ answer letters are sensitive to label and order.

Actions:

1. For each MCQ, run the model on 3-5 random option permutations.
2. Map each predicted letter back to the original option.
3. Vote by option text, not by letter.
4. If option-text votes tie, use the finalizer or verifier confidence.

Why: this directly attacks option-order and label bias. It is also easy to
evaluate on public MCQ rows.

Expected effort: medium.

Expected payoff: medium, likely higher on MCQ than free-form.

### Priority 5: Adaptive Test-Time Compute

Problem: the full rerank path spends extra compute on rows where it is not
needed and can damage correct answers.

Actions:

Use extra compute only when one of these triggers fires:

- raw/finalized answer is missing;
- generated trace hit the token limit;
- finalizer slot count mismatches expected slot count;
- 1024 and 2048 candidates disagree;
- MCQ option-permutation votes disagree;
- verifier confidence is low;
- problem has 3+ answer slots.

Why: adaptive compute follows the test-time scaling literature while keeping
runtime manageable.

Expected effort: medium.

Expected payoff: high per token spent.

## Experiment Matrix

Run these in order. Stop if a method fails on the frozen validation split.

| ID | Experiment | Main Question | Success Criteria |
|---|---|---|---|
| E0 | Frozen stratified eval | Are we measuring real gains? | One command reports stable split metrics |
| E1 | Slot-aware finalizer retry | Does it fix multi-slot misses? | Better 3+ slot accuracy with no MCQ regression |
| E2 | MCQ option permutation | Does label/order bias matter here? | Higher MCQ accuracy on held-out MCQ rows |
| E3 | Candidate diversity sweep | Are candidates different enough? | Higher oracle accuracy than current 1024/2048 pair |
| E4 | Verifier LoRA | Can Qwen select better candidates? | Selected accuracy beats both single best candidate and current reranker |
| E5 | RFT solver LoRA | Can correct traces improve reasoning? | Solver+finalizer beats base+finalizer |
| E6 | Adaptive policy | Can we combine gains safely? | Better public-weighted score than Method 1 |
| E7 | Dense-reward GRPO | Does RL add beyond RFT? | Beats RFT solver or verifier selection |

The most important diagnostic is oracle accuracy: if any candidate in the set is
correct. If oracle accuracy is low, generation needs improvement. If oracle
accuracy is high but selected accuracy is low, selection/verifier is the
bottleneck.

## Candidate Diversity Plan

The current 1024/2048 pair is not diverse enough. Candidate generation should
vary multiple axes:

- Prompt: `final_box_only`, `slot_exact`, `symbolic_strict`,
  `exact_conservative`, `mcq_concise_compute`.
- Token budget: 768, 1024, 1536, 2048, maybe 3072 for selected hard rows.
- Temperature: 0.2 for conservative, 0.6 for diverse, 0.9 for hard rows.
- MCQ option order: original plus 3-5 permutations.
- Assistant mode: thinking for solving, direct for finalization only.

Do not increase diversity blindly. Track:

- number of unique answer keys per row;
- oracle accuracy;
- selected accuracy;
- token-limit hit rate;
- malformed answer rate;
- slot-count mismatch rate.

## Verifier Design

Recommended first verifier format:

```text
System:
You are Qwen verifying a math competition answer. Do not solve from scratch
unless needed. Check whether the candidate final answer satisfies the problem.
Output exactly \boxed{yes} or \boxed{no}.

User:
Problem:
...

Candidate final answer:
...

Candidate reasoning tail:
...

Is the candidate final answer correct?
```

Training labels:

- positive: candidate scored correct by `judger.py`;
- negative: candidate scored incorrect by `judger.py`;
- hard negatives: candidates with plausible but wrong normalized answer keys;
- hard positives: correct candidates with unusual but equivalent symbolic forms.

Inference:

- For each candidate, generate yes/no with low temperature.
- Prefer candidates marked yes.
- If multiple yes candidates exist, prefer the one with:
  - valid answer key;
  - correct slot count;
  - no token-limit hit;
  - higher vote count;
  - longer source reasoning only as a final tie-breaker.

This is intentionally simpler than a full PRM. It tests whether verifier
selection helps before investing in step-level reward modeling.

## Slot-Aware Finalizer Design

The finalizer should see explicit metadata:

```text
Expected final answer format:
- This is a free-form problem.
- Number of [ANS] slots: 4.
- Output exactly 4 ordered sub-answers inside one \boxed{}.
- Separate sub-answers with commas.
```

Retry rule:

1. Run normal finalizer.
2. Extract answer.
3. If slot count mismatches, rerun with a stricter prompt and lower
   temperature.
4. If it still mismatches, use the best candidate whose extracted key has the
   correct count, even if it came from raw reasoning.

This is a narrow, high-leverage fix because it targets the format failures
already observed in the report.

## MCQ Design

Recommended MCQ pipeline:

1. Ask Qwen to solve and produce the answer text or target expression, not only
   the letter.
2. Ask Qwen/finalizer to map the answer text to an option letter.
3. Repeat with shuffled options.
4. Vote by original option index.
5. If disagreement remains, run verifier on the top two option texts.

This avoids relying only on the model's prior over letters such as A/B/C/D.

## What Not To Prioritize

- More one-off prompt variants on the first 20 rows. The evidence already shows
  overfitting.
- Full MCTS before building a verifier. Search without a useful scorer is
  expensive random sampling.
- GRPO before RFT. The current reward is too sparse; RFT is simpler and better
  matched to labeled public data.
- vLLM as an accuracy improvement. vLLM is a speed tool. It enables larger
  candidate sweeps, but it does not directly improve answer selection.
- Longer generation for every row. More tokens can help hard rows, but it also
  increases unfinished traces and runtime. Use adaptive budgets.

## Recommended Next Implementation Order

1. Add a frozen validation/evaluation script for stratified metrics and oracle
   candidate accuracy.
2. Add slot-aware finalizer retry.
3. Add MCQ option-permutation voting.
4. Add candidate diversity sweeps and measure oracle accuracy.
5. Train a yes/no verifier LoRA from generated public candidates.
6. Train a rejection-sampled solver LoRA from correct Qwen traces.
7. Combine with adaptive routing.
8. Revisit GRPO only after the above baselines are exhausted.

## Source Index

- Chain-of-thought prompting: https://arxiv.org/abs/2201.11903
- Self-consistency: https://arxiv.org/abs/2203.11171
- Universal self-consistency: https://arxiv.org/abs/2311.17311
- Training verifiers: https://arxiv.org/abs/2110.14168
- Process supervision: https://arxiv.org/abs/2305.20050
- Qwen2.5-Math self-improvement: https://arxiv.org/abs/2409.12122
- Rejection sampling fine-tuning: https://arxiv.org/abs/2308.01825
- DeepSeekMath and GRPO: https://arxiv.org/abs/2402.03300
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- rStar-Math: https://arxiv.org/abs/2501.04519
- Self-Refine: https://arxiv.org/abs/2303.17651
- MetaMath: https://arxiv.org/abs/2309.12284
- WizardMath: https://arxiv.org/abs/2308.09583
- MAmmoTH: https://arxiv.org/abs/2309.05653
- MathScale: https://arxiv.org/abs/2403.02884
- OpenMathInstruct-1: https://arxiv.org/abs/2402.10176
- MCQ selector robustness: https://arxiv.org/abs/2309.03882
- MCQ option order sensitivity: https://arxiv.org/abs/2308.11483
- Text-based MCQ selection: https://arxiv.org/abs/2404.08382
- Qwen vLLM deployment notes: https://qwen.readthedocs.io/en/stable/deployment/vllm.html

