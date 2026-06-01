# CSE 151B Competition Inference Submission

This branch contains the inference code and submitted LoRA weights needed to reproduce the final submission pipeline.

## Hardware and Runtime

- GPU type used: 4x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97 GB VRAM each.
- Main generation/finalization run: approximately 20-24 hours including the 8k/16k/32k reruns and CSV validation.
- Base model: `Qwen/Qwen3-4B-Thinking-2507`.

## Weights

The two submitted PEFT LoRA adapters are uploaded to Hugging Face Hub and are the defaults used by `run_inference()`:

```text
wren88/cse151b-chungers-frq-finalizer-lora
wren88/cse151b-chungers-rebuilt-answer-lora
```

They are loaded on top of `Qwen/Qwen3-4B-Thinking-2507`. If the base model or adapters are not already cached, Hugging Face/vLLM will download them on first run.

Local copies are also included in this repository with Git LFS:

```text
models/frq_finalizer_lora
models/rebuilt_answer_lora
```

To force local adapter loading, pass those paths with `--frq-adapter` and `--rebuilt-adapter`, or pass them as the matching `run_inference()` keyword arguments.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Put the competition data at:

```text
data/private.jsonl
```

## Run Inference

Python entry point:

```python
from run_inference import run_inference

run_inference(
    data_path="data/private.jsonl",
    output_csv="submissions/final_submission.csv",
)
```

CLI equivalent:

```bash
python run_inference.py \
  --data data/private.jsonl \
  --output submissions/final_submission.csv \
  --tensor-parallel-size 4
```

The output CSV has the required columns:

```text
id,response
```

## Pipeline

`run_inference()` performs the full pipeline end-to-end:

1. Load `Qwen/Qwen3-4B-Thinking-2507`.
2. Generate raw reasoning with an 8k/16k/32k token ladder for rows that hit the token limit.
3. Finalize free-response rows with `wren88/cse151b-chungers-frq-finalizer-lora`.
4. Finalize multiple-choice rows with `wren88/cse151b-chungers-rebuilt-answer-lora`.
5. Normalize responses into boxed final answers.
6. Validate one non-empty prediction per input id and write the submission CSV.

Final hyperparameters:

```text
raw generation temperature: 0.6
raw generation top_p: 0.95
raw generation top_k: 20
raw token ladder: 8192, 16384, 32768
finalizer temperature: 0.1
finalizer top_p: 0.9
finalizer max_new_tokens: 64
finalizer retry max_new_tokens: 128
vLLM dtype: float16
```
