"""Compatibility exports for the modular generation and retrieval pipeline."""

from generation.answer import (
    CitationAudit,
    _citation_claims,
    _composite_confidence,
    _confidence_report,
    _insufficient_evidence_response,
    _judge_citation_claim,
    audit_citations,
    ask,
    build_agent,
    build_chain,
    get_prompt_template,
    reset_memory,
    retrieval_confidence,
    score_answer_completeness,
    verify_citations,
)
from retrieval.context import format_context as format_docs
from retrieval.search import reciprocal_rank_fusion

__all__ = [
    "CitationAudit",
    "ask",
    "audit_citations",
    "build_agent",
    "build_chain",
    "format_docs",
    "get_prompt_template",
    "reciprocal_rank_fusion",
    "reset_memory",
    "retrieval_confidence",
    "score_answer_completeness",
    "verify_citations",
]
