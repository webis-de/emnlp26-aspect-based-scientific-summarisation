"""
This module defines the input and output data structures for the fact score evaluation agents. 
These structures are used to standardize the communication between different components of the evaluation pipeline, such as claim decomposition and evidence retrieval.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional

# ----------------------------------------------------------------#
# --- Fact Score ---
class ClaimDecompositionInput(BaseModel):
    summary_text: str


class ClaimDecompositionOutput(BaseModel):
    claims: List[str] = Field(
        ...,
        description="A list of atomic factual claims extracted from the summary.",
    )
# ----------------------------------------------------------------#
