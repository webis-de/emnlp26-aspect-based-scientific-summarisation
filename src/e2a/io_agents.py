"""
This module defines the input and output data models for each agent in the 2A2S pipeline:
1. Extractor
2. Writer
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ----------------------------------------------------------------#
# --- Extractor ---
class ExtractorInput(BaseModel):
    document_text: str
    aspect_name: str
    aspect_definition: Optional[str] = None
    extraction_cues: Optional[List[str]] = None
    correction_instruction: Optional[str] = None
    max_evidence_items: Optional[int] = None
    max_span_chars: Optional[int] = None

class ExtractorOutput(BaseModel):
    evidence_set: List[str] = Field(
        ...,
        description="A list of extracted evidence spans (in document order).",
    )

# ----------------------------------------------------------------#
# --- Writer ---
class WriterInput(BaseModel):
    aspect_name: str
    summary_length: str  # e.g., "200 words"
    evidence_set: List[str]  # Extractor output spans
    aspect_definition: Optional[str] = None
    acceptance_criteria: Optional[List[str]] = None
    correction_instruction: Optional[str] = None
    prior_summary: Optional[str] = None
    

class WriterOutput(BaseModel):
    summary_text: str = Field(
        ..., 
        description="The final synthesized summary."
    )
