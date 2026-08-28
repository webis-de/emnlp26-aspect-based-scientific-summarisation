**Self-Refine is a single-pass three-stage summarization pipeline** — **Generate → Suggest → Refine** — not a recursive loop.

## 1) What the workflow does

Given a scientific text and one target aspect, the agent first generates an initial aspect-focused summary, then asks the **same model** to critique that draft with **2–3 concrete, non-generic improvement suggestions**, and finally starts a **fresh conversation** in which it asks the model to rewrite the summary using the scientific text, the original draft, and the suggestions. The prompts are designed so the suggestions stay grounded in the source (scientific) text, while the final rewrite is explicitly conditioned on both the first draft and the critique.  

The summary length is **dataset-dependent** and should come from `SUMMARY_TARGETS` in `src/config.py`. For the current datasets that means `aclsum` targets 1 sentence / 25 words, `facetsum` targets 2 sentences / 50 words, and `pmc` targets 4 sentences / 75 words.

## 2) Minimal Python-level contract

Use one function like this conceptually:

```python
def self_refine(source_text: str, aspect: str, llm_chat) -> dict:
    """
    source_text: source text
    aspect: target aspect
    llm_chat: callable that takes chat messages and returns assistant text
    returns:
        {
            "initial_summary": str,
            "suggestions": str,
            "refined_summary": str,
        }
    }
```

The workflow should return **all three artifacts**: the initial draft, the suggestion list, and the refined draft. The paper’s prompt structure makes these three outputs the natural intermediate states of the method. 

## 3) Exact control flow the coding agent should implement

### Stage A — GENERATION

Create a chat with:

1. one **system** message defining the model as an abstractive summarizer focused on one aspect,
2. one **user** message asking for a short summary within the configured sentence and word budget based on the source text and focused on `{aspect}`. 

Call the LLM once and store the assistant output as:

```python
initial_summary: str
```

### Stage B — SUGGESTION

**Do not restart the conversation yet.** Continue the same chat from Stage A, so the model still has access to:

* the original system instruction,
* the original user prompt containing the source text,
* the assistant’s generated summary. 

Append a new **user** message asking for:

* a **short list of 2–3 suggestions**,
* to make the generated summary **more concise**,
* and **more focused on the target aspect**,
* with suggestions grounded in the **original source text** and the **generated summary**,
* and explicitly **not generic**. 

Call the LLM once and store the assistant output as:

```python
suggestions: str
```

In practice, the paper later analyzes these suggestions as usually appearing in a **structured list format**, often emphasizing things like being more concise, more focused on the aspect, more specific, or merging/removing less relevant content. For implementation, you can treat this as **raw text** and pass it through unchanged to Stage C.

### Stage C — REFINE

Now **restart the conversation**. This is explicit in Table 10. Start a fresh chat with:

1. the same or equivalent **system** message as in Stage A,
2. one **user** message that includes:

   * the original source text,
   * the aspect,
   * the full original summary,
   * the full suggestion text,
    * and an instruction to produce an improved version within the configured sentence and word budget. 

Call the LLM once and store the assistant output as:

```python
refined_summary: str
```

Return all three fields. 

## 4) Prompt templates for implementation

Below are **faithful paraphrase templates** for coding. They preserve the method in the paper, but they are not a verbatim copy.

### Stage A prompt

```python
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
```

This matches the paper’s Self-Refine generation stage: an aspect-baed, short summary grounded in the source text. 

### Stage B prompt

```python
SUGGEST_USER_TEMPLATE = """
Give 2-3 concrete suggestions to improve the generated summary so that it becomes more concise and more focused on the aspect "{aspect}".

Base the suggestions on the original source text and the generated summary.
Do not give generic advice.
""".strip()
```

For this stage, keep the Stage A chat history intact and append this as a new user turn after the model’s first draft. In the paper, this is the “Suggest” step inside the same conversation; the reset happens only afterward. 

### Stage C prompt

```python
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
```

This captures the paper’s refine stage: fresh chat, same summarizer role, and a user prompt that re-injects the source text, the first draft, and the critique. 

## 5) Message structure the coding agent should use

A faithful chat orchestration looks like this:

```python
# Stage A
messages_ab = [
    {"role": "system", "content": GEN_SYSTEM},
    {"role": "user", "content": GEN_USER_TEMPLATE.format(aspect=aspect, source_text=source_text)},
]
initial_summary = llm_chat(messages_ab)

# Stage B (same conversation)
messages_ab.append({"role": "assistant", "content": initial_summary})
messages_ab.append({
    "role": "user",
    "content": SUGGEST_USER_TEMPLATE.format(aspect=aspect),
})
suggestions = llm_chat(messages_ab)

# Stage C (restart conversation)
messages_c = [
    {"role": "system", "content": REFINE_SYSTEM},
    {
        "role": "user",
        "content": REFINE_USER_TEMPLATE.format(
            aspect=aspect,
            source_text=source_text,
            initial_summary=initial_summary,
            suggestions=suggestions,
        ),
    },
]
refined_summary = llm_chat(messages_c)
```

This is the closest coding translation of Table 10: the first two calls share context, and the third call uses a fresh conversation with the earlier artifacts embedded directly in the prompt. 

## 6) Output handling rules

The paper does **not** define any strict machine-readable schema for Self-Refine outputs. So for re-creation:

* treat `initial_summary` as plain text,
* treat `suggestions` as plain text,
* treat `refined_summary` as plain text,
* and pass the suggestion text through unchanged into the refine prompt.

A reasonable implementation detail is to do only light cleanup:

* strip whitespace,
* optionally trim enclosing quotes,
* optionally enforce the configured word cap after generation if your target model occasionally exceeds it.
  That cleanup is an implementation convenience; it is not explicitly specified in the paper. 

## 7) Important “do not accidentally change the method” notes

Do **not** turn this into:

* a multi-iteration self-improvement loop,
* a verifier-based pipeline,
* a sentence-level critique system,
* or the E2A extract-then-abstract JSON workflow.

In this paper, Self-Refine is only the three-stage **Generate → Suggest → Refine** chain with a conversation reset before the final rewrite. The E2A method is separate and uses extraction-specific prompting and structured output.

## 8) Short implementation summary for the coding agent

Implement a function that:

1. generates an initial aspect-focused summary from source text,
2. asks the same model in the same chat for 2–3 grounded revision suggestions,
3. starts a fresh chat and asks the model to rewrite the summary using the source text, the original draft, and the suggestion list,
4. returns `initial_summary`, `suggestions`, and `refined_summary`.

I can also turn this into a **drop-in Python implementation** using either the OpenAI chat format or Hugging Face transformers chat templates.
