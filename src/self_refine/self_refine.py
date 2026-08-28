"""Core Self-Refine workflow implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from language_engine import LanguageEngine

try:
    from .prompts import (
        GEN_SYSTEM,
        GEN_USER_TEMPLATE,
        REFINE_SYSTEM,
        REFINE_USER_TEMPLATE,
        SUGGEST_USER_TEMPLATE,
        build_summary_target_instruction,
        clean_text,
        enforce_word_limit,
    )
except ImportError:
    from prompts import (
        GEN_SYSTEM,
        GEN_USER_TEMPLATE,
        REFINE_SYSTEM,
        REFINE_USER_TEMPLATE,
        SUGGEST_USER_TEMPLATE,
        build_summary_target_instruction,
        clean_text,
        enforce_word_limit,
    )


def flatten_source_text(source_text: object) -> str:
    """Convert long-context structured source text into prompt-ready plain text."""
    if isinstance(source_text, str):
        return source_text.strip()
    if not isinstance(source_text, dict):
        return str(source_text).strip()

    parts: List[str] = []
    for section_title, section_text in source_text.items():
        header = str(section_title).strip()
        content = "" if section_text is None else str(section_text).strip()
        if header and content:
            parts.append(f"## {header}\n{content}")
        elif content:
            parts.append(content)
    return "\n\n".join(parts).strip()


@dataclass(frozen=True)
class ChatGenerationResult:
    text: str
    num_rollouts: int
    input_tokens: int
    output_tokens: int


def _count_tokens_from_text(engine: LanguageEngine, text: str) -> int:
    if not text:
        return 0
    tokenized = engine.tokenizer(text, add_special_tokens=False)
    return len(tokenized.get("input_ids", []))


def llm_chat_from_messages(
    engine: LanguageEngine,
    messages: List[Dict[str, str]],
    enable_thinking: bool = False,
) -> ChatGenerationResult:
    """Run a full chat history through the existing LanguageEngine instance."""
    full_prompt = engine.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    outputs = engine.model.generate([full_prompt], engine.sampling_params, use_tqdm=False)

    request_output = outputs[0]
    completion_outputs = request_output.outputs
    if not completion_outputs:
        raise ValueError("LanguageEngine returned no completions.")

    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    if prompt_token_ids is None:
        input_tokens = _count_tokens_from_text(engine, full_prompt)
    else:
        input_tokens = len(prompt_token_ids)

    output_tokens = 0
    for completion in completion_outputs:
        completion_token_ids = getattr(completion, "token_ids", None)
        if completion_token_ids is None:
            output_tokens += _count_tokens_from_text(engine, completion.text)
        else:
            output_tokens += len(completion_token_ids)

    return ChatGenerationResult(
        text=completion_outputs[0].text.strip(),
        num_rollouts=len(completion_outputs),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def self_refine(
    source_text: object,
    aspect: str,
    engine: LanguageEngine,
    target_sentences: int = 2,
    max_words: int = 100,
    enable_thinking: bool = False,
) -> Dict[str, str | int]:
    """Run the single-pass Self-Refine workflow for one source/aspect pair."""
    source_text_str = flatten_source_text(source_text)
    summary_target_instruction = build_summary_target_instruction(
        target_sentences=target_sentences,
        max_words=max_words,
    )

    messages_ab = [
        {"role": "system", "content": GEN_SYSTEM},
        {
            "role": "user",
            "content": GEN_USER_TEMPLATE.format(
                aspect=aspect,
                source_text=source_text_str,
                summary_target_instruction=summary_target_instruction,
            ),
        },
    ]
    initial_result = llm_chat_from_messages(
        engine,
        messages_ab,
        enable_thinking=enable_thinking,
    )
    initial_summary = enforce_word_limit(clean_text(initial_result.text), max_words=max_words)

    messages_ab.append({"role": "assistant", "content": initial_summary})
    messages_ab.append(
        {
            "role": "user",
            "content": SUGGEST_USER_TEMPLATE.format(aspect=aspect),
        }
    )
    suggestions_result = llm_chat_from_messages(
        engine,
        messages_ab,
        enable_thinking=enable_thinking,
    )
    suggestions = clean_text(suggestions_result.text)

    messages_c = [
        {"role": "system", "content": REFINE_SYSTEM},
        {
            "role": "user",
            "content": REFINE_USER_TEMPLATE.format(
                aspect=aspect,
                source_text=source_text_str,
                initial_summary=initial_summary,
                suggestions=suggestions,
                summary_target_instruction=summary_target_instruction,
            ),
        },
    ]
    refined_result = llm_chat_from_messages(
        engine,
        messages_c,
        enable_thinking=enable_thinking,
    )
    refined_summary = enforce_word_limit(clean_text(refined_result.text), max_words=max_words)

    usage_results = (initial_result, suggestions_result, refined_result)

    return {
        "initial_summary": initial_summary,
        "suggestions": suggestions,
        "refined_summary": refined_summary,
        "num_rollouts": sum(result.num_rollouts for result in usage_results),
        "input_tokens": sum(result.input_tokens for result in usage_results),
        "output_tokens": sum(result.output_tokens for result in usage_results),
    }