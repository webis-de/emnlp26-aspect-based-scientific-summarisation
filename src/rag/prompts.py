import io_agents as i_o


class WriterPrompts:
    SYSTEM = (
        "You are an expert scientific editor, you are given a scientific text and tasked to write "
        "a summary focused on a specific aspect. "
        "You use the same wording and tone used in the source text, which is scientific and formal. "
        "Write concise, faithful aspect-focused summaries."
    )

    USER_TEMPLATE = """\
# CONTEXT
You will be provided with a scientific text and a specific aspect to focus on.

# TASK
Given the scientific text below and a focused aspect, which is {aspect_name}, write a short summary using your own words.
The summary needs to be a coherent paragraph and should include the major points.
Write in free form, avoid bullet points or numbered lists.
The summary should focus on the provided aspect only, contain only information about the aspect, and avoid adding irrelevant sentences or your own opinions and suggestions.

# INPUT
Note: the source text below is not the complete paper — it is a selection of passages retrieved as most relevant to the aspect.
FOCUSED ASPECT: {aspect_name}
{summary_length_line}
SOURCE TEXT:
{d_pruned}

# OUTPUT
Return a valid JSON object:
{{
    "summary_text": "Your single-paragraph summary..."
}}"""

    @staticmethod
    def render(data: i_o.WriterInput) -> tuple[str, str]:
        summary_length_line = f"SUMMARY TARGET LENGTH: {data.summary_length}"
        user_prompt = WriterPrompts.USER_TEMPLATE.format(
            aspect_name=data.aspect_name,
            summary_length_line=summary_length_line,
            d_pruned=data.d_pruned,
        )
        return WriterPrompts.SYSTEM, user_prompt
