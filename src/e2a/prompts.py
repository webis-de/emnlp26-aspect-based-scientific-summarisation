from jinja2 import Template
from textwrap import dedent
import io_agents as i_o

class ExtractorPrompts:
    # PERSONA
    SYSTEM = (
        "You are an Information Extractor specializing in scientific literature. "
        "You excel in identifying and extracting verbatim evidence that is necessary for summarizing specific aspects of research papers. "
        "You value precision; You never alter quotes or add interpretation. "
        "You produce structured JSON OUTPUT only."
    )
    
    # THE TEMPLATE
    USER_TEMPLATE = dedent("""
    ### CONTEXT
    You are given a scientific document and a specific aspect. 
                           
    ### TASK
    Your task is to extract verbatim evidence from the document that is relevant to the specified aspect and would be useful for writing a summary about that aspect.
      
    ### INPUTS
    Target Aspect: 
    {{ aspect_name }}                                            
                           
    Source Document:
    {{ document_text }}

    ### INSTRUCTIONS
    1. Find spans of text that contain claims or data relevant to the aspect. 
    2. Extract verbatim text spans, do not paraphrase or summarize. 
    3. Each extracted span should be self-contained and informative on its own.
    4. Avoid extracting very short or very long spans; aim for concise, meaningful excerpts that capture key points.
    5. The spans should be sorted in the order they appear in the document to preserve context.
    6. Return at most {{ max_evidence_items }} evidence items.
    7. Keep each span short (prefer 1-2 sentences, max about {{ max_span_chars }} characters).
    8. Avoid references, equations, and low-signal boilerplate.
    9. Output valid JSON only.
    
    ### OUTPUT FORMAT
    Return a valid JSON object matching the ExtractorOutput schema:
    {
        "evidence_set": [
            "Exact quote 1...",
            "Exact quote 2..."
        ]
    }
    """).strip()

    @staticmethod
    def render(data: i_o.ExtractorInput) -> tuple[str, str]:
        t = Template(ExtractorPrompts.USER_TEMPLATE)
        payload = data.model_dump()
        payload["max_evidence_items"] = payload.get("max_evidence_items") or 50
        payload["max_span_chars"] = payload.get("max_span_chars") or 500
        return ExtractorPrompts.SYSTEM, t.render(**payload)


class WriterPrompts:
    # PERSONA
    SYSTEM = (
        "You are an expert scientific assistant and editor. "
        "Your expertise lies in synthesizing information from multiple pieces of evidence to create concise, accurate summaries focused on specific aspects of scientific research. "
        "You use the same wording and tone used in the source text, which is scientific and formal. Write concise, faithful aspect-focused summaries."
    )

    USER_TEMPLATE = dedent("""
    ### CONTEXT
    You are given a set of verbatim evidence extracted from a scientific document, all related to a the aspect "{{ aspect_name }}".

    ### TASK
    Write a short aspect-focused summary with your own words for the following aspect: "{{ aspect_name }}" . 
    Use ONLY the provided evidence set.
    The summary needs to be a coherent paragraph and should include the major points. 

    ### EVIDENCE SET (ORDERED)
    {{ evidence_set }}

    ### SUMMARY LENGTH
    {{ summary_length }}

    ### INSTRUCTIONS
    1. The summary should focus on the provided aspect only. 
    2. Use ONLY the evidence set above.
    3. Write in free form, avoid bullet points or numbered lists. 
    4. Avoid adding irrelevant sentences or your own opinions and suggestions.
    5. Respect the summary length instruction.

    ### OUTPUT FORMAT
    Return a valid JSON object matching the WriterOutput schema:
    {
        "summary_text": "Your single-paragraph summary..."
    }
    """).strip()

    @staticmethod
    def render(data: i_o.WriterInput) -> tuple[str, str]:
        t = Template(WriterPrompts.USER_TEMPLATE)
        return WriterPrompts.SYSTEM, t.render(**data.model_dump())



# # test example
#if __name__ == "__main__":

    # Extractor
    # sample_input = i_o.ExtractorInput(
    #     text_type="abstract",
    #     aspect_name="Methodology",
    #     document_text="This study employs a mixed-methods approach to investigate the effects of..."
    #     #aspect_definition="Methodology refers to the detailed procedures and techniques employed in the research study, including data collection, experimental design, and analysis methods.",
    #     #extraction_cues=["Section titled 'Methods'", "Phrases like 'we conducted', 'data collection involved'"]
    # )
    # system_prompt, user_prompt = ExtractorPrompts.render(sample_input)
    # print("SYSTEM PROMPT:\n", system_prompt)
    # print("\nUSER PROMPT:\n", user_prompt)
