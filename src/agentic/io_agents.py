"""
This module defines the input and output data models for each agent in the 2A2S pipeline:
1. Planner
2. Extractor
3. Writer
4. Verifier
5. Router
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional

# ----------------------------------------------------------------#
# --- Planner ---
class PlannerInput(BaseModel):
    aspect_name: str
    document_text: str  
    text_type: str 

class PlannerOutput(BaseModel):
    aspect_definition: str = Field(
        ..., 
        description="A brief definition of what this aspect means in the context of this specific text type."
    )
    extraction_cues: List[str] = Field(
        ..., 
        description="A list of concrete semantic triggers or keywords to look for in the text."
    )
    acceptance_criteria: List[str] = Field(
        ..., 
        description="A checklist of items that a perfect summary of this aspect must contain."
    )

# ----------------------------------------------------------------#
# --- Extractor ---
# Sub-Module 
class EvidenceItem(BaseModel):
    """
    Represents a single atomic unit of evidence extracted from the text.
    """
    span: str = Field(
        ..., 
        description="The exact sentence(s) containing the key claim or fact."
    )
    section: Optional[str] = Field(
        None,
        description="Optional: the document section (e.g., 'Methods', 'Results') this span came from."
    )

class ExtractorInput(BaseModel):
    document_text: str
    aspect_name: str
    source_type: Optional[str] = Field(
        None,
        description="Document source/type label if available."
    )
    source_type_description: Optional[str] = Field(
        None,
        description="Short description of the source/type label if available."
    )
    
    # -- Ablation ---
    # If Planner is removed, these will be None.
    aspect_definition: Optional[str] = None
    extraction_cues: Optional[List[str]] = None

    # --- Feedback ---
    # If this is a retry loop (judgment='revise_evidence'), this field contains
    # the specific instruction on what was missed previously.
    correction_instruction: Optional[str] = Field(
        None, 
        description="Feedback from the Verifier regarding missing evidence."
    )

class ExtractorOutput(BaseModel):
    evidence_set: List[EvidenceItem] = Field(
        ..., 
        description="A list of extracted spans (in document order). Each item is the textual span and an optional section label."
    )

# ----------------------------------------------------------------#
# --- Writer ---
class WriterInput(BaseModel):
    aspect_name: str
    summary_length: str  # e.g., "200 words"

    evidence_set: List['EvidenceItem']  # Forward reference to Extractor's output
    
    # --- Ablation ---
    # If Planner is removed, these will be None.
    aspect_definition: Optional[str] = None
    acceptance_criteria: Optional[List[str]] = Field(
        None,
        description="The checklist of requirements the summary must satisfy."
    )

    # --- Feedback ---
    # If this is a retry loop (judgment='revise_summary'), this field contains
    # the specific instruction on what to fix.
    correction_instruction: Optional[str] = Field(
        None, 
        description="Critique from the Verifier on the previous draft."
    )

    # --- Revision Context ---
    # If this is a revise_summary loop, the latest prior summary can be provided
    # to help the Writer revise with feedback.
    prior_summary: Optional[str] = Field(
        None,
        description="Most recent summary draft to revise, provided only on revise_summary."
    )

class WriterOutput(BaseModel):
    summary_text: str = Field(
        ..., 
        description="The final synthesized summary."
    )

# ----------------------------------------------------------------#
# --- Verifier ---
class VerifierInput(BaseModel):
    aspect_name: str
    summary_text: str
    evidence_set: List['EvidenceItem'] # Forward reference to Extractor
    
    # --- Ablation ---
    # If Planner is removed, these will be None.
    aspect_definition: Optional[str] = None
    acceptance_criteria: Optional[List[str]] = None

class VerifierOutput(BaseModel):
    judgment: Literal['accept', 'revise_summary', 'revise_evidence']
    
    issue_description: str = Field(
        ..., 
        description="An explanation of the main issue found with the summary or evidence."
    )

    correction_instruction: Optional[str] = Field(
        None,
        description="Specific instructions for the next agent to fix the issue."
    )
    
    action_hint: Literal['none', 'rewrite_with_same_evidence', 'extract_additional_evidence']

# ----------------------------------------------------------------#
# --- Router ---
class RouterInput(BaseModel):
    verifier_judgment: Literal['accept', 'revise_summary', 'revise_evidence']
    action_hint: Literal['none', 'rewrite_with_same_evidence', 'extract_additional_evidence']

    # The Verifier's explanation of what went wrong
    issue_description: Optional[str] = None

    # The Router needs to receive the specific instruction to pass it on.
    correction_instruction: Optional[str] = None
    
    # Loop Guards 
    current_iteration: int
    max_iterations: int

class RouterOutput(BaseModel):
    next_role: Literal[
        'stop',       # End the process (Success or Max Retries Reached)
        'writer',     # Loop back to Agent 3
        'extractor'   # Loop back to Agent 2
    ]
    
    # The Mailman delivers the message here
    correction_instruction: Optional[str] = Field(
        None, 
        description="The instruction to be passed to the next_role's input."
    )
    
    termination_reason: Optional[Literal['success', 'max_retries_exceeded']] = None
# ----------------------------------------------------------------#
