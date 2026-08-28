"""Prompt templates and text helpers for the Self-Refine pipeline."""

GEN_SYSTEM = """
You are an abstractive summarizer. Your task is to summarize a scientific text focusing on one specific aspect.
""".strip()

GEN_USER_TEMPLATE = """
Write a short summary about the aspect "{aspect}" using only the source text below.

Target length: {summary_target_instruction}.

Reflections:
{source_text}

Summary:
""".strip()

SUGGEST_USER_TEMPLATE = """
Give 2-3 concrete suggestions to improve the generated summary so that it becomes more concise and more focused on the aspect "{aspect}".

Base the suggestions on the original source text and the generated summary.
Do not give generic advice.
""".strip()

REFINE_SYSTEM = GEN_SYSTEM

REFINE_USER_TEMPLATE = """
Improve the short summary below using the suggestions.

The revised version must stay focused on the aspect "{aspect}" based on the source text below.

Target length: {summary_target_instruction}.

Reflections:
{source_text}

Original summary:
{initial_summary}

Suggestions for improvement:
{suggestions}

Refined summary:
""".strip()


def build_summary_target_instruction(target_sentences: int, max_words: int) -> str:
    """Render dataset-specific sentence and word guidance for prompts."""
    if target_sentences < 1:
        raise ValueError("target_sentences must be at least 1")
    if max_words < 1:
        raise ValueError("max_words must be at least 1")

    sentence_label = "sentence" if target_sentences == 1 else "sentences"
    return f"{target_sentences} {sentence_label} and no more than {max_words} words"


def clean_text(text: str) -> str:
    """Apply light cleanup without changing content semantics."""
    cleaned = text.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def enforce_word_limit(text: str, max_words: int = 100) -> str:
    """Trim only when the model exceeds the expected word budget."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()