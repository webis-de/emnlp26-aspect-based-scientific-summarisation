from jinja2 import Template
from textwrap import dedent
import io_agents as i_o

class ClaimDecompositionPrompts:
    SYSTEM = (
        "You are a factual claim decomposition assistant. "
        "You excellently extract atomic factual claims from scientific summaries, ensuring they are explicit, non-overlapping, and free of inference. "
        "Output valid JSON only."
    )

    USER_TEMPLATE = dedent("""
    ### TASK
    Decompose the summary into atomic factual claims.

    ### SUMMARY
    {{ summary_text }}

    ### INSTRUCTIONS
    1. Extract only factual claims explicitly stated in the summary.
    2. Keep claims atomic and non-overlapping where possible.
    3. Do not invent, infer, or add external information.
    4. Remove duplicates and avoid empty claims.
    5. If no factual claims exist, return an empty list.

    ### OUTPUT FORMAT
    Return a valid JSON object matching:
    {
        "claims": ["claim 1", "claim 2"]
    }
    """).strip()

    @staticmethod
    def render(data: i_o.ClaimDecompositionInput) -> tuple[str, str]:
        t = Template(ClaimDecompositionPrompts.USER_TEMPLATE)
        return ClaimDecompositionPrompts.SYSTEM, t.render(**data.model_dump())


