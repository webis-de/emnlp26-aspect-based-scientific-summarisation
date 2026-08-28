from dataclasses import dataclass, field
from typing import Dict, Optional, Any

@dataclass
class AspectSummaryDatapoint:
    """Dataclass for aspect-based summary gold datapoints."""
    source_text: str
    source_type: str
    summary: Dict[str, str]
    doc_id : Optional[str] = None
    full_text: str | Dict[str, str] | None = None

@dataclass(slots=True)
class SummarizationResult:
	"""Dataclass container for summarization results."""

	summary: str | Dict[str, str] # Can be a string or a dict for aspect-based summaries. String is used for heuristic-based summaries.
	metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GoldDatapoint:
    source_text: str
    aspect_summary: str
    aspect_name: str
    source_type: str # abstract, abstract-introduction-conclusion, introduction-conclusion, full-text
    context_size: str # short, long
    dataset: str # pmc, facetsum, aclsum, scholarsum

@dataclass
class GeneratedDatapoint:
    generated_aspect_summary: str 
    gold_aspect_summary: str
    aspect_name: str
    source_text: str
    source_type: str
    context_size: str
    dataset: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScoredGeneratedDatapoint:
    generated_aspect_summary: str 
    gold_aspect_summary: str
    aspect_name: str
    source_text: str
    source_type: str
    context_size: str
    dataset: str
    rouge1: float
    rouge2: float
    rougeL: float
    rougeLsum: float
    bertscore: float
    g_eval: float
    fact_checker: float
    metadata: Dict[str, Any] = field(default_factory=dict)