# Added by Codex: prompt construction helpers; not part of the original starter repository.

from __future__ import annotations

from typing import Any


DEFAULT_MATH_SYSTEM = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside "
    "a single \\boxed{}, e.g. \\boxed{3, 7}."
)

DEFAULT_MCQ_SYSTEM = (
    "You are an expert mathematician. Read the problem and answer choices, "
    "then select the single best answer. Put only the final option letter "
    "inside \\boxed{}, e.g. \\boxed{C}."
)


def format_options(options: list[str]) -> str:
    labels = [chr(65 + idx) for idx in range(len(options))]
    return "\n".join(f"{label}. {option.strip()}" for label, option in zip(labels, options))


def build_messages(item: dict[str, Any], prompt_config: dict[str, Any] | None = None) -> list[dict[str, str]]:
    prompt_config = prompt_config or {}
    options = item.get("options") or []
    if options:
        system = prompt_config.get("mcq_system", DEFAULT_MCQ_SYSTEM)
        user = f"{item['question']}\n\nOptions:\n{format_options(options)}"
        suffix = prompt_config.get("mcq_user_suffix") or prompt_config.get("user_suffix")
    else:
        system = prompt_config.get("math_system", DEFAULT_MATH_SYSTEM)
        user = item["question"]
        suffix = prompt_config.get("math_user_suffix") or prompt_config.get("user_suffix")

    if suffix:
        user = f"{user}\n\n{suffix}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_prompt_text(tokenizer: Any, item: dict[str, Any], prompt_config: dict[str, Any] | None = None) -> str:
    prompt_config = prompt_config or {}
    text = tokenizer.apply_chat_template(
        build_messages(item, prompt_config),
        tokenize=False,
        add_generation_prompt=True,
    )
    if prompt_config.get("assistant_mode") == "direct":
        forced_think = "<|im_start|>assistant\n<think>\n"
        direct = "<|im_start|>assistant\n"
        if text.endswith(forced_think):
            text = text[: -len(forced_think)] + direct
    assistant_prefix = prompt_config.get("assistant_prefix")
    if assistant_prefix:
        text += assistant_prefix
    return text
