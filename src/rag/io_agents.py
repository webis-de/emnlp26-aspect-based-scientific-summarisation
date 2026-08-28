"""Input/output data models for the RAG pipeline."""

from pydantic import BaseModel, Field
from typing import List, Optional


class RetrieverInput(BaseModel):
    document_text: str
    aspect_name: str
    chunk_size: int
    top_k_chunks: int
    word_budget: int


class RetrieverOutput(BaseModel):
    d_pruned: str
    num_chunks: int
    num_sentences_retained: int
    num_words_retained: int
    chunk_scores: Optional[List[float]] = None


class WriterInput(BaseModel):
    aspect_name: str
    summary_length: str
    d_pruned: str
    correction_instruction: Optional[str] = None


class WriterOutput(BaseModel):
    summary_text: str = Field(..., description="The final synthesized summary.")
