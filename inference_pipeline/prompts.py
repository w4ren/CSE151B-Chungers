from __future__ import annotations

from typing import Any

from inference_pipeline.formatting import format_contract, format_options, is_mcq


RAW_SYSTEM = (
    "You are an expert mathematician. Solve the problem carefully. "
    "End with only the final answer in \\boxed{}."
)

FINALIZER_SYSTEM = (
    "You are a math answer finalizer. Use the problem and the previous solver trace "
    "to extract the final answer. Do not explain. Obey the output schema exactly."
)


def direct_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    forced_think = "<|im_start|>assistant\n<think>\n"
    if text.endswith(forced_think):
        text = text[: -len(forced_think)] + "<|im_start|>assistant\n"
    return text


def problem_text(item: dict[str, Any]) -> str:
    text = str(item["question"])
    if is_mcq(item):
        text += "\n\nOptions:\n" + format_options(item.get("options") or [])
    return text


def build_raw_prompt(tokenizer: Any, item: dict[str, Any]) -> str:
    user = (
        "Problem:\n"
        f"{problem_text(item)}\n\n"
        "Required final-answer schema:\n"
        f"{format_contract(item)}"
    )
    return direct_chat_prompt(tokenizer, [{"role": "system", "content": RAW_SYSTEM}, {"role": "user", "content": user}])


def build_finalizer_prompt(
    tokenizer: Any,
    item: dict[str, Any],
    trace: str,
    *,
    max_trace_chars: int,
    retry_reason: str | None = None,
    first_attempt: str | None = None,
) -> str:
    clipped_trace = str(trace or "")[-max_trace_chars:]
    retry = ""
    if retry_reason:
        retry = (
            "The previous finalizer attempt was rejected because "
            f"{retry_reason}.\nPrevious attempt:\n{first_attempt or ''}\n\n"
        )
    user = (
        "Problem:\n"
        f"{problem_text(item)}\n\n"
        "Required final-answer schema:\n"
        f"{format_contract(item)}\n\n"
        f"{retry}"
        "Previous solver trace:\n"
        f"{clipped_trace}\n\n"
        "Final answer only:"
    )
    return direct_chat_prompt(tokenizer, [{"role": "system", "content": FINALIZER_SYSTEM}, {"role": "user", "content": user}])
